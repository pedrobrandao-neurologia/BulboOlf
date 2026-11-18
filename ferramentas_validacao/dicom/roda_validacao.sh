#!/usr/bin/env bash
# Valida a conversão DICOM -> NIfTI do BulboOlf contra a verdade sintética e contra o dcm2niix.
# Requisitos: python3 com pydicom, numpy, nibabel, python-gdcm; dcm2niix; node >= 20.
# Uso (nesta pasta): bash roda_validacao.sh /caminho/para/pasta_de_trabalho
set -euo pipefail
W=${1:?informe uma pasta de trabalho}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W" && cd "$W"
python3 "$H/gera_dicom_sintetico.py" dcmtest
python3 "$H/comprime_jpeg_gdcm.py" dcmtest/exp_le dcmtest/jpeg_sv1 sv1
python3 "$H/comprime_jpeg_gdcm.py" dcmtest/exp_le dcmtest/jpeg_p14 p14
python3 "$H/comprime_jpeg_gdcm.py" dcmtest/signed_slope dcmtest/jpeg_signed sv1
python3 "$H/gera_exame_misto.py"
for d in exp_le imp_le exp_be signed_slope rle deflate enh; do mkdir -p "d2n/$d"; dcm2niix -z y -f "%f" -o "d2n/$d" "dcmtest/$d" >/dev/null || echo "dcm2niix não converteu $d"; done
for d in exp_le imp_le exp_be signed_slope rle deflate enh jpeg_sv1 jpeg_p14 jpeg_signed; do node "$H/converte_dicom.mjs" "dcmtest/$d" "js_$d.nii.gz" | tail -1; done
python3 "$H/compara_conversao.py"
node "$H/converte_dicom.mjs" mix js_mix.nii.gz
