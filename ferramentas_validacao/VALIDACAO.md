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

## Conversão DICOM → NIfTI (etapa antes do pipeline)

Volume sintético 72×80×50 com voxel 0,75×0,8×0,7 mm em orientação **oblíqua** (12° e −7°), fatias gravadas em ordem
aleatória e arquivos sem extensão. Comparação do NIfTI gerado por `js/dicom.js` com a verdade e com o `dcm2niix`
(v1.0.20261029), após `nibabel.as_closest_canonical`:

| Codificação | Δ dados × verdade | Δ affine × verdade | × dcm2niix |
| --- | --- | --- | --- |
| VR explícita LE / implícita LE / explícita BE | 0 | ≤ 3×10⁻⁷ | dados idênticos, Δ affine 3×10⁻⁵ |
| int16 com sinal + RescaleSlope 1,5 / Intercept 3 | 0 | ≤ 3×10⁻⁷ | idem |
| RLE Lossless | 0 | ≤ 3×10⁻⁷ | idem |
| Deflate | 0 | ≤ 3×10⁻⁷ | dcm2niix não lê deflate |
| Enhanced MR multiframe (quadros embaralhados, RescaleSlope 2 no grupo compartilhado) | 0 | ≤ 3×10⁻⁷ | dados idênticos; dcm2niix desloca 0,55 mm neste arquivo sintético (sem DimensionIndexSequence) |
| JPEG sem perdas SV1 e processo 14 (GDCM), inclusive com sinal | 0 | ≤ 3×10⁻⁷ | — |

Exame misto (localizador, T1 MPRAGE, T2 TSE 2D, T2 SPACE 0,8 iso, MPR derivada, FLAIR 3D, T2 3D com fatia faltando):
a sugestão automática escolhe a T2 SPACE 0,8 iso; localizador e série com fatia faltando ficam bloqueados.
O NIfTI convertido passa pela reorientação LAS, normalização e reamostragem do pipeline sem alteração de código.

Scripts em `dicom/` (`bash roda_validacao.sh <pasta_de_trabalho>`; requer pydicom, nibabel, python-gdcm, dcm2niix, node).
