# BulboOlf — volumetria do bulbo olfatório por RM (PWA)

Aplicativo web progressivo (PWA) que roda **no navegador, sem enviar imagens a servidores**:

1. **Segmentação automática** do bulbo olfatório (BO) esquerdo e direito — porte fiel para JavaScript/ONNX do pipeline de
   Estrada S, Lu R, Diers K, et al. *Automated olfactory bulb segmentation on high resolutional T2-weighted MRI.*
   NeuroImage 2021;242:118464. doi:10.1016/j.neuroimage.2021.118464 — código e pesos originais:
   https://github.com/Deep-MI/olf-bulb-segmentation (Apache-2.0).
2. **Revisão / planimetria manual** (pincel com rótulos 1 = esquerdo, 2 = direito, borracha, preenchimento, desfazer).
3. **Profundidade do sulco olfatório** no plano tangente posterior aos globos oculares (3 cliques por lado).
4. Tabela de sujeitos (grupo AF/Controle, idade, sexo, escore olfativo, QC, notas) com exportação **CSV**,
   sessão em **JSON** e segmentações em **NIfTI**. Script R incluído (`R/analise_bulbo_olfatorio.R`).

## Entrada exigida pelo método automático

- **T2 3D isotrópico de 0,7–0,8 mm, cérebro inteiro (cabeça inteira)**, em NIfTI-1 (`.nii` ou `.nii.gz`).
  Converta DICOM com `dcm2niix -z y`.
- Imagens com outra resolução são reamostradas para 0,8 mm (como no original), mas o método **não foi validado**
  fora de 0,7–0,8 mm. Para T2 coronal 2D (fatias de 1–2 mm) use a planimetria manual.
- Não use imagens desfiguradas (*defaced*) ou com extração de crânio: podem cortar o BO.

## Como rodar

O PWA precisa ser servido por HTTP (não abre por `file://`).

- **GitHub Pages:** suba o conteúdo desta pasta num repositório e ative Pages (branch `main`, pasta raiz).
  O *service worker* (`sw.js`) ativa o isolamento de origem, necessário para o WASM com várias threads.
- **Local:** `python3 -m http.server 8000` nesta pasta e abra `http://localhost:8000`.
- Use Chrome ou Edge atualizados. Com WebGPU disponível (Mac com Apple Silicon, GPUs recentes) a inferência usa a GPU;
  sem WebGPU, usa a CPU (WASM). Rode o **Autoteste** uma vez em cada computador.
- **Baixar modelos (offline)** guarda as 15 redes (55 MB) e o runtime (29 MB) no cache do navegador.

## Fluxo sugerido (AF × controles)

1. Adicione os NIfTI (o ID é tirado do nome do arquivo; grupo é sugerido se o nome tiver `AF`/`ctrl`).
2. Escolha **12 modelos** (padrão do artigo) e **localização automática** → *Processar pendentes*.
3. Para cada sujeito: confira a segmentação na *grade do pipeline (0,8 mm)*; corrija com *Editar / planimetria* e
   clique *Registrar volumes*; meça o sulco; marque QC (`ok`, `revisar`, `excluir`).
4. *Exportar CSV* → `Rscript R/analise_bulbo_olfatorio.R bulbo_olfatorio_AAAA-MM-DD.csv`.
5. Recomendado: planimetria manual cega de um subconjunto por 2 avaliadores para relatar concordância local.

## Saídas (iguais às do pipeline original)

`OB_E/D/total_vol_mm3` (volume = nº de voxels × volume do voxel da grade do pipeline), `nvox`, intensidade média
normalizada, **entropia** e **CV entre os modelos do ensemble** (incerteza), distância localização–segmentação e
alertas (`OB esquerdo/direito não encontrado`, `Nenhum OB aparente`). Os autores do método recomendam inspeção visual
quando o volume total < 20 mm³.

## Fidelidade ao original — o que foi verificado

- As 15 redes PyTorch foram exportadas para ONNX trocando MaxPool/MaxUnpool por operações equivalentes (argmax por
  janela 2×2, primeira ocorrência) e a autoatenção por blocos (mesmo resultado, menos memória).
  Diferença PyTorch × ONNX ≤ 2×10⁻⁵ em logits para entradas sem empates.
- Toda a lógica (reorientação LAS, normalização robusta de intensidade, reamostragem linear, localização por 3
  planos, recorte 96³, ensemble, componentes conexos E/D, estatísticas) foi comparada com o **código Python original**
  usando modelos-substitutos: segmentação idêntica voxel a voxel (0 voxels discordantes) e estatísticas idênticas, em
  imagem 0,88 mm RAS (reamostrada) e 0,8 mm LIA int16 (sem reamostragem).
- ONNX Runtime Web (navegador) × ONNX Runtime Python em fatias reais: probabilidades com diferença ≤ 2×10⁻⁷.
  Diferenças de logit aparecem só em pixels de fundo (zero), onde empates no max-pooling são desfeitos de forma
  diferente por cada motor — o mesmo ocorre entre CPU e GPU no PyTorch.

## Licenças

Pipeline e pesos: Apache-2.0 (DZNE / Deep-MI) — ver `NOTICE.md`. ONNX Runtime Web: MIT. NiiVue: BSD-2-Clause.
Ferramenta de pesquisa; não usar para diagnóstico.
