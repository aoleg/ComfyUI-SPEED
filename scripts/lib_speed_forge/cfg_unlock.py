"""Keep the negative prompt box usable at CFG 1.0 while SPEED's RAW composition stage is on.

Adapted from Neo-Krea2-NAG's ``lib_krea2_nag/cfg_unlock.py``. Forge greys the negative
prompt out at CFG 1.0, correctly, since ``setup_conds`` ignores it there. The RAW
composition stage encodes it itself and applies it with real CFG on the coarse SPEED
segments, so the box has to come back while the stage is on, without editing Forge.

There is exactly one mechanism to displace. ``modules/ui.py`` defines

    def use_cfg(val):
        return gr.skip() if val is None else gr.update(interactive=(val > 1.0))

and binds it right after each tab's CFG slider is built:

    cfg_scale.change(fn=use_cfg, inputs=[cfg_scale], outputs=[toprow.negative_prompt], ...)

``fn=use_cfg`` resolves ``modules.ui.use_cfg`` when that line runs, so substituting the
function before the binding is enough; no second listener competes for the event.

The flag is per tab and is "SPEED on and the RAW stage on". Gradio hands ``use_cfg``
only the slider value, so the tab is chosen by which variant is in place when each
binding runs, and the variants are swapped between the bindings by ``after_component``
callbacks. Where this differs from NAG's module, it is so the two can be installed
together and still chain, each variant calling the function it replaced whenever its
own flag is off, so the box unlocks when either extension wants it:

* The txt2img variant is installed at the first component built after ``before_ui``,
  not in ``before_ui`` itself, so it always wraps whatever another extension
  substituted there (NAG does) instead of being captured by it.
* The img2img variant wraps whatever is in place when ``img2img_cfg_scale`` is built,
  and is re-applied at ``img2img_image_cfg_scale`` (built after the slider and before
  its binding) if another extension's own img2img swap replaced it in between.
* The hires CFG slider (``txt2img_hr_cfg``) binds the same function to the hires
  negative prompt. The RAW stage never runs on the hires pass, so that binding gets the
  function this module wrapped, not the unlocking variant.
* ``before_ui`` unwraps the variants of the previous build, so a UI reload starts from
  the chain as it was. If another extension's ``before_ui`` runs first on a reload, it
  can capture the previous build's variant as its own original, and that variant's
  chain then leads back to itself through the other extension's stale variant (NAG's
  module keeps one global original, so this is a real loop, not a hypothetical). A
  re-entered variant therefore answers with Forge's rule instead of recursing: the box
  unlocks too eagerly in that case (this module's other-tab flag joins the chain), and
  never locks wrongly or raises.

Ticking a checkbox fires no CFG change, so the script also wires both checkboxes to push
the update; that half lives in ``scripts/speed_forge.py``, where the components are.
"""

from __future__ import annotations

import threading
from typing import Any

import gradio as gr

from modules import script_callbacks

TABS = ("txt2img", "img2img")

_enabled = {tab: False for tab in TABS}
_installed = False
_base_use_cfg = None
"""The ``use_cfg`` this module wrapped at install time, for the untick update."""

_local = threading.local()


def set_enabled(tab: str, value: bool):
    if tab in _enabled:
        _enabled[tab] = bool(value)


def is_enabled(tab: str) -> bool:
    return _enabled.get(tab, False)


def unlock_update(tab: str, speed_enabled: bool, stage_enabled: bool, cfg_scale: float | None = None):
    """The update to push at the negative prompt box when either checkbox changes.

    Off hands the box back to the rule it would follow without this module rather than
    locking it: the run may be at CFG > 1, where the box is legitimately live, or another
    extension may want it open. Without the CFG value there is no way to know, and
    ``gr.skip()`` (leave it until the slider next moves) is the honest answer.
    """
    enabled = bool(speed_enabled) and bool(stage_enabled)
    set_enabled(tab, enabled)

    if enabled:
        return gr.update(interactive=True)

    if cfg_scale is None or _base_use_cfg is None:
        return gr.skip()

    return _base_use_cfg(cfg_scale)


def _tab_of(fn) -> str | None:
    return getattr(fn, "_speed_unlock_tab", None)


def _forge_rule(val):
    """``modules.ui.use_cfg`` as Forge defines it."""
    return gr.skip() if val is None else gr.update(interactive=(val > 1.0))


def _make_use_cfg(tab: str, previous):
    def use_cfg(val):
        if _enabled.get(tab):
            return gr.update(interactive=True)
        if getattr(_local, "active", False):
            # Re-entered: a stale chain after a reload loops back here.
            return _forge_rule(val)
        _local.active = True
        try:
            return previous(val)
        finally:
            _local.active = False

    use_cfg._speed_unlock_tab = tab
    use_cfg._speed_unlock_previous = previous
    return use_cfg


def _unwrap(fn):
    while _tab_of(fn) is not None:
        fn = fn._speed_unlock_previous
    return fn


def _forge_ui():
    try:
        from modules import ui as forge_ui
    except Exception:
        return None
    return forge_ui if callable(getattr(forge_ui, "use_cfg", None)) else None


def _before_ui():
    global _installed
    _installed = False
    forge_ui = _forge_ui()
    if forge_ui is not None and _tab_of(forge_ui.use_cfg) is not None:
        forge_ui.use_cfg = _unwrap(forge_ui.use_cfg)


def _install(forge_ui):
    global _installed, _base_use_cfg
    _installed = True
    previous = _unwrap(forge_ui.use_cfg)
    _base_use_cfg = previous
    forge_ui.use_cfg = _make_use_cfg("txt2img", previous)


def _on_component(component: Any, **kwargs):
    forge_ui = _forge_ui()
    if forge_ui is None:
        return

    if not _installed:
        _install(forge_ui)

    elem_id = getattr(component, "elem_id", None)
    current = forge_ui.use_cfg

    if elem_id == "txt2img_hr_cfg":
        if _tab_of(current) == "txt2img":
            forge_ui.use_cfg = current._speed_unlock_previous
    elif elem_id in ("img2img_cfg_scale", "img2img_image_cfg_scale"):
        if _tab_of(current) == "img2img":
            return
        previous = current._speed_unlock_previous if _tab_of(current) == "txt2img" else current
        forge_ui.use_cfg = _make_use_cfg("img2img", previous)


def register():
    script_callbacks.on_before_ui(_before_ui)
    script_callbacks.on_after_component(_on_component)
