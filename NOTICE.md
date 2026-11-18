# Avisos de terceiros

## Pipeline e pesos de segmentação do bulbo olfatório
Copyright 2021 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases (DZNE).
Licenciado sob a Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
Origem: https://github.com/Deep-MI/olf-bulb-segmentation
Modificações neste projeto: os pesos (`*.pkl`) foram convertidos para ONNX (`models/*.onnx`); MaxPool2d/MaxUnpool2d
reescritos como argmax por janela e a camada Self_Attn calculada em blocos, com equivalência numérica verificada; o
pré e pós-processamento (`conform`, `OBNet`, `image_utils`, `stats`) foi reescrito em JavaScript (`js/pipeline.js`).
Ao publicar resultados, cite: Estrada S, et al. NeuroImage 2021;242:118464. doi:10.1016/j.neuroimage.2021.118464

## ONNX Runtime Web (vendor/ort*.mjs, vendor/*.wasm)
Copyright (c) Microsoft Corporation. MIT License.

## NiiVue (vendor/niivue.js)
Copyright (c) NiiVue contributors. BSD 2-Clause License. https://github.com/niivue/niivue
