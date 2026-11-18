# Validação do porte (18/11/2026)

Imagem de teste: `chris_t2.nii.gz` (T2 3D, 0,88 mm isotrópico, cabeça inteira; repositório público niivue-images).

| Verificação | Resultado |
| --- | --- |
| PyTorch (pesos originais) × ONNX, por modelo, entrada aleatória | máx. Δ logit ≤ 2×10⁻⁵ (15 modelos) — ver `export_report.json` |
| Lógica JS × código Python original, modelos-substitutos, imagem 0,88 mm RAS (reamostrada) | 0 voxels discordantes; volumes, intensidades, entropia, CV, centros e distância idênticos |
| Idem, imagem 0,8 mm LIA int16 com scl_slope (sem reamostragem; ramo "L" do clean_seg) | 0 voxels discordantes; estatísticas idênticas |
| Navegador (Chrome, WASM) × Node, modelos-substitutos | 0 voxels discordantes |
| Localização automática real (código original + ONNX), chris_t2 | centro em (102, 198, 62) na grade 0,8 mm = sulco/fossa olfatória; 30 min em 2 vCPU |
| Segmentação real, 3 modelos: JS (ONNX Runtime Web) × Python original (ONNX Runtime) | 0 voxels discordantes; estatísticas idênticas |
| Segmentação real, 3 modelos: PyTorch original × ONNX | 0 voxels discordantes; máx. Δ prob. 6×10⁻⁶ |
| Autoteste no navegador, WASM e WebGPU (lavapipe), fatias reais com bulbo | Δ prob. ≤ 5×10⁻⁶; concordância de classe 100% |

Volumes em chris_t2 (12 modelos): E 8,7 mm³, D 8,2 mm³, total 16,9 mm³ — abaixo do limiar de 20 mm³ em que os autores
recomendam inspeção visual, com CV entre modelos 0,28; a imagem de teste não tem contraste suficiente na fossa olfatória
para servir de referência de volume. Ela serviu apenas para conferir a equivalência numérica.

Scripts: `export_onnx.py` (conversão + checagem), `make_standin.py` (modelos-substitutos), `py_ref_driver.py` e
`py_ref_seg.py` (rodam o código original com ONNX Runtime), `node_pipeline_test.mjs` (pipeline JS em Node), `compare.py`.
