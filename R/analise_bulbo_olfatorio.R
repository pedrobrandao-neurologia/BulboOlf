# analise_bulbo_olfatorio.R
# Análise do CSV exportado pelo BulboOlf (AF × controles).
# Uso: Rscript analise_bulbo_olfatorio.R bulbo_olfatorio_AAAA-MM-DD.csv
#
# Variável principal: volume total do bulbo (mm³). Usa o volume revisado/manual quando existir;
# caso contrário, o volume automático. Sujeitos com qc == "excluir" são removidos.

args <- commandArgs(trailingOnly = TRUE)
arq <- if (length(args)) args[1] else file.choose()
txt <- readLines(arq, encoding = "UTF-8", warn = FALSE)
txt[1] <- sub("^\xef\xbb\xbf", "", txt[1], useBytes = TRUE)  # remove BOM
d <- read.csv(text = txt, stringsAsFactors = FALSE, encoding = "UTF-8")

d <- subset(d, qc != "excluir")
d$grupo <- factor(d$grupo, levels = c("Controle", "AF"))
d$sexo <- factor(d$sexo)
d$idade <- suppressWarnings(as.numeric(d$idade))
d$vol_total <- ifelse(!is.na(d$OB_total_vol_revisado_mm3), d$OB_total_vol_revisado_mm3, d$OB_total_vol_mm3)
d$vol_E <- ifelse(!is.na(d$OB_E_vol_revisado_mm3), d$OB_E_vol_revisado_mm3, d$OB_E_vol_mm3)
d$vol_D <- ifelse(!is.na(d$OB_D_vol_revisado_mm3), d$OB_D_vol_revisado_mm3, d$OB_D_vol_mm3)
d$fonte_volume <- ifelse(!is.na(d$OB_total_vol_revisado_mm3), "revisado", "automatico")

cat("\n== Sujeitos por grupo e fonte do volume ==\n")
print(table(d$grupo, d$fonte_volume, useNA = "ifany"))

cat("\n== Alertas do pipeline ==\n")
print(d[d$alerta != "" & !is.na(d$alerta), c("sub_id", "grupo", "alerta", "OB_total_vol_mm3")])
cat("\nVolumes totais < 20 mm3 (desempenho do pipeline cai abaixo disso; revisar):\n")
print(d[!is.na(d$vol_total) & d$vol_total < 20, c("sub_id", "grupo", "vol_total")])

desc <- function(x) if (all(is.na(x))) "sem dados" else sprintf("%.1f ± %.1f (mediana %.1f; %.1f–%.1f)", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE),
                            median(x, na.rm = TRUE), min(x, na.rm = TRUE), max(x, na.rm = TRUE))
cat("\n== Descritiva (mm³): média ± DP (mediana; mín–máx) ==\n")
for (v in c("vol_E", "vol_D", "vol_total", "sulco_E_mm", "sulco_D_mm")) {
  if (!v %in% names(d) || all(is.na(d[[v]]))) next
  cat(sprintf("%-12s", v), paste(levels(d$grupo), sapply(levels(d$grupo), function(g) desc(d[[v]][d$grupo == g])), sep = ": ", collapse = " | "), "\n")
}

cat("\n== Comparação AF × controles (volume total) ==\n")
print(wilcox.test(vol_total ~ grupo, data = d, exact = FALSE))
cov <- c("grupo", if (sum(!is.na(d$idade)) > 3) "idade", if (length(unique(na.omit(d$sexo))) > 1) "sexo")
f <- as.formula(paste("vol_total ~", paste(cov, collapse = " + ")))
m <- lm(f, data = d)
cat("\nModelo linear:", deparse(f), "\n"); print(summary(m)); print(confint(m))
# tamanho de efeito (d de Cohen, sem ajuste)
g <- split(d$vol_total, d$grupo); sp <- sqrt(((length(na.omit(g[[1]])) - 1) * var(g[[1]], na.rm = TRUE) + (length(na.omit(g[[2]])) - 1) * var(g[[2]], na.rm = TRUE)) / (length(na.omit(g[[1]])) + length(na.omit(g[[2]])) - 2))
cat(sprintf("\nd de Cohen (AF − controle): %.2f\n", (mean(g[[2]], na.rm = TRUE) - mean(g[[1]], na.rm = TRUE)) / sp))

if ("escore_olfativo" %in% names(d) && sum(!is.na(d$escore_olfativo)) > 5) {
  cat("\n== Volume total × escore olfativo (Spearman) ==\n")
  print(cor.test(d$vol_total, d$escore_olfativo, method = "spearman", exact = FALSE))
}

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  p <- ggplot(d, aes(grupo, vol_total)) + geom_boxplot(outlier.shape = NA, width = .5) +
    geom_jitter(width = .12, height = 0, alpha = .7) + labs(x = NULL, y = "Volume total do bulbo olfatório (mm³)") + theme_minimal(base_size = 12)
  ggsave(sub("\\.csv$", "_boxplot.png", arq), p, width = 4, height = 4.5, dpi = 200)
  cat("\nFigura salva:", sub("\\.csv$", "_boxplot.png", arq), "\n")
}
