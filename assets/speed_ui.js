// [SPEED] The "Spectrum A" / "Spectrum Beta" inputs should only be visible when
// "Model Preset" is set to "custom" -- they are ignored for every other preset.
//
// SwarmUI's built-in `DependNonDefault` can't express that: it only supports
// "show when the other param differs from *its own* default" (i.e. "!= flux"),
// which would show these two for every non-flux preset, not just "custom".
// So visibility is handled here instead, via the `visible_controlled` /
// `hideParamCallbacks` extension point genpage/gentab/params.js exposes for
// exactly this kind of custom show/hide logic.
(function() {
    const PRESET_ID = 'speedmodelpreset';
    const CUSTOM_FIELD_IDS = ['speedspectruma', 'speedspectrumbeta'];

    function customFieldBoxes() {
        return CUSTOM_FIELD_IDS
            .map(id => document.getElementById(`input_${id}`))
            .filter(elem => elem)
            .map(elem => findParentOfClass(elem, 'auto-input'))
            .filter(box => box);
    }

    postParamBuildSteps.push(() => {
        for (let box of customFieldBoxes()) {
            box.dataset.visible_controlled = 'true';
        }
    });

    hideParamCallbacks.push(() => {
        let presetElem = document.getElementById(`input_${PRESET_ID}`);
        let isCustom = presetElem && getInputVal(presetElem) == 'custom';
        for (let box of customFieldBoxes()) {
            box.style.display = isCustom ? '' : 'none';
        }
    });
})();
