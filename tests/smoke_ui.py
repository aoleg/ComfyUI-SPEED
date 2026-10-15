"""UI smoke test: build the Forge script's gradio UI with the REAL gradio from the
Forge Neo venv (fake webui modules, real components). Run from the repo root with
the venv python, e.g.:

    <forge>/venv/Scripts/python.exe tests/smoke_ui.py

Catches gradio API misuse the MagicMock harness in test_forge_script.py cannot
see, and checks that both checkboxes push an update to the negative prompt box.
``InputAccordion`` is a small stand-in built on real gradio components, so Forge's
own InputAccordion path (and the unlock in the browser) is still only exercised live.
"""

import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402


class InputAccordion:
    """Forge's InputAccordion reduced to its shape: a hidden checkbox (the
    input value) plus an accordion that holds the section's components.
    ``with InputAccordion(...) as x`` yields the checkbox, as Forge's does
    (there the class is itself the checkbox). Not a gr.Checkbox subclass:
    stock gradio writes a .pyi stub for every Component subclass it sees."""

    def __init__(self, value=None, **kwargs):
        label = kwargs.get("label", "Accordion")
        self.checkbox = gr.Checkbox(value=value, label=label, visible=False)
        self.accordion = gr.Accordion(label=label, open=bool(value))

    def __enter__(self):
        self.accordion.__enter__()
        return self.checkbox

    def __exit__(self, *args):
        return self.accordion.__exit__(*args)


_app_started = []

modules = types.ModuleType("modules")
modules.__path__ = []
scripts_mod = types.ModuleType("modules.scripts")
scripts_mod.Script = type("Script", (), {})
scripts_mod.AlwaysVisible = object()
scripts_mod.basedir = lambda: REPO
scripts_mod.scripts_data = []
errors_mod = types.ModuleType("modules.errors")
errors_mod.report = lambda msg, **kw: print(f"errors.report: {msg}")
ui_components_mod = types.ModuleType("modules.ui_components")
ui_components_mod.InputAccordion = InputAccordion
callbacks_mod = types.ModuleType("modules.script_callbacks")
callbacks_mod.on_before_ui = lambda fn: None
callbacks_mod.on_after_component = lambda fn: None
callbacks_mod.on_app_started = _app_started.append
for name, mod in (("modules", modules), ("modules.scripts", scripts_mod),
                  ("modules.errors", errors_mod),
                  ("modules.ui_components", ui_components_mod),
                  ("modules.script_callbacks", callbacks_mod)):
    sys.modules[name] = mod
    if name != "modules":
        setattr(modules, name.split(".", 1)[1], mod)

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "speed_forge_smoke", os.path.join(REPO, "scripts", "speed_forge.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def build(is_img2img):
    tab = "img2img" if is_img2img else "txt2img"
    with gr.Blocks() as demo:
        negative = gr.Textbox(label="Negative prompt", elem_id=f"{tab}_neg_prompt")
        cfg = gr.Slider(minimum=1.0, maximum=24.0, step=0.5, value=1.0,
                        label="CFG Scale", elem_id=f"{tab}_cfg_scale")
        script = mod.SpeedForgeScript()
        for component in (negative, cfg):
            script.after_component(component)
        components = script.ui(is_img2img)

    assert len(components) == 18, f"expected 18 components, got {len(components)}"
    enabled, diversity = components[0], components[13]
    ref_steps, raw_cfg, pattern, bake = components[14:18]
    assert (ref_steps.minimum, ref_steps.maximum, ref_steps.value) == (28, 52, 28)
    assert (raw_cfg.minimum, raw_cfg.maximum, raw_cfg.value) == (1.0, 10.0, 4.5)
    assert pattern.value == "turbo"
    assert bake.value is True

    pushes = []
    for fn in demo.fns.values() if isinstance(demo.fns, dict) else demo.fns:
        outputs = [getattr(o, "_id", o) for o in fn.outputs]
        if negative._id in outputs:
            targets = {t[0] if isinstance(t, (tuple, list)) else t for t in fn.targets}
            pushes.append((targets, [getattr(i, "_id", i) for i in fn.inputs]))
    expected_inputs = [enabled._id, diversity._id, cfg._id]
    for checkbox in (enabled, diversity):
        assert any(checkbox._id in targets and inputs == expected_inputs
                   for targets, inputs in pushes), (tab, pushes)
    print(f"OK {tab}: gradio {gr.__version__}, {len(components)} components, "
          f"{len(pushes)} negative-prompt updates wired")


build(False)
build(True)
assert len(_app_started) == 2, _app_started

unlock = mod.cfg_unlock
_app_started[0](None, None)
assert unlock.is_enabled("txt2img") is False
assert unlock.unlock_update("txt2img", True, True, 1.0) == gr.update(interactive=True)
assert unlock.is_enabled("txt2img") is True and unlock.is_enabled("img2img") is False
print("OK unlock_update")
