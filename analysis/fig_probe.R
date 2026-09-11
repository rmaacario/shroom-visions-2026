# Figuras da sonda de representation shift, para o paper do SHROOM-Visions.
# Dados em probe_categories.csv / probe_layers.csv / probe_overall.csv,
# gerados por export_fig_data.py. Sem título: a explicação vai na legenda do LaTeX.
#
# Paleta: por omissão escala de cinza, que sobrevive a impressão e fotocópia --
# é a convenção do venue (CHOMPS 2025 usa dois acentos no máximo; AILS-NTUA
# reserva cor para destacar spans). Para ligar um acento, ponha ACCENT_ON <- TRUE.
#
# Funciona tanto com Rscript quanto colado no console do RStudio: os CSVs são
# procurados em vários lugares prováveis (ver find_data), então não é preciso
# dar setwd(). Se os seus dados estiverem noutro sítio, preencha DATA_DIR.
library(ggplot2)

DATA_DIR   <- ""          # opcional: caminho onde estão os probe_*.csv
INK        <- "#1A1A1A"
GREY       <- "#9A9A9A"   # so para reguas e linhas de referencia
BAR        <- "#A6BEDC"   # azul das caixas do proxy na Figura 1
ACCENT     <- "#CE7332"   # laranja HalMark: o mesmo de \hal{} e do anel da Figura 1
ACCENT_ON  <- TRUE  # TRUE destaca as duas categorias dominantes
COL_W      <- 3.15        # largura de uma coluna ACL, em polegadas

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", a, value = TRUE)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  of <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(of)) return(dirname(normalizePath(of)))
  getwd()   # console do RStudio: cai aqui, daí o find_data abaixo
}

CANDS <- unique(Filter(nzchar, c(
  DATA_DIR, script_dir(), getwd(),
  file.path(path.expand("~"), "Downloads"),
  file.path(path.expand("~"), "Downloads", "shroom-visions"))))

find_data <- function(fname) {
  for (d in CANDS) {
    p <- file.path(d, fname)
    if (file.exists(p)) return(normalizePath(p))
  }
  stop(sprintf(paste0("nao encontrei %s. Procurei em:\n  %s\n",
                      "Gere os CSVs com  python3 export_fig_data.py  ",
                      "na pasta que contem feats/, ou preencha DATA_DIR."),
               fname, paste(CANDS, collapse = "\n  ")), call. = FALSE)
}

f_cats  <- find_data("probe_categories.csv")
DIR     <- dirname(f_cats)          # figuras saem ao lado dos dados
cat("dados em:", DIR, "\n")

save_fig <- function(stem, plot, w, h) {
  f <- file.path(DIR, paste0(stem, ".pdf"))
  cairo_ok <- isTRUE(capabilities("cairo")) &&
    !inherits(suppressWarnings(try({ tmp <- tempfile(fileext = ".pdf")
                                     cairo_pdf(tmp); dev.off(); unlink(tmp) },
                                   silent = TRUE)), "try-error")
  if (cairo_ok) ggsave(f, plot, width = w, height = h, units = "in", device = cairo_pdf)
  else          ggsave(f, plot, width = w, height = h, units = "in")
  cat("ok:", f, "\n"); invisible(f)
}

base_theme <- theme_minimal(base_size = 8, base_family = "serif") +
  theme(panel.grid.minor   = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = "#DDDDDD", linewidth = 0.25),
        axis.line.x        = element_line(colour = "#666666", linewidth = 0.25),
        axis.ticks         = element_blank(),
        axis.text          = element_text(colour = "#444444", size = 7),
        axis.title         = element_text(colour = "#222222", size = 8),
        plot.margin        = margin(2, 6, 2, 2))

# ── Figura 1: AUROC por categoria, quatro linguas ───────────────────────────
cats <- read.csv(f_cats)
en   <- cats[cats$lang == "en", ]
en   <- en[order(-en$share), ]
lbl  <- sub("mischaracterization", "mischaracteriz.", en$category)
keys <- sprintf("%s  %.0f%%", lbl, 100 * en$share)
cats$label <- factor(keys[match(cats$category, en$category)], levels = rev(keys))
cats$lang  <- factor(toupper(cats$lang), levels = c("EN", "FR", "IT", "ZH"))
winner     <- en$category[which.max(en$auroc)]
cats$hi    <- if (ACCENT_ON) cats$category == winner else FALSE

lo <- min(cats$auroc) - 0.03; hi <- max(cats$auroc) + 0.03

p1 <- ggplot(cats, aes(x = auroc, y = label, shape = lang, colour = hi)) +
  geom_errorbar(aes(xmin = auroc - sd, xmax = auroc + sd), orientation = "y",
                width = 0, colour = "#BBBBBB", linewidth = 0.3) +
  geom_point(size = 1.6, fill = "white", stroke = 0.5) +
  scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = ACCENT), guide = "none") +
  scale_shape_manual(values = c(EN = 16, FR = 21, IT = 24, ZH = 22), name = NULL) +
  coord_cartesian(xlim = c(lo, hi), ylim = c(0.7, nrow(en) + 0.3), clip = "off") +
  labs(x = "probe AUROC", y = NULL) +
  base_theme +
  theme(legend.position = "bottom", legend.margin = margin(-6, 0, 0, 0),
        legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6.6, colour = "#444444"))
save_fig("fig_categories", p1, COL_W, 1.85)

# ── Figura 2: AUROC por profundidade ────────────────────────────────────────
lay  <- read.csv(find_data("probe_layers.csv"))
keep <- c("early", "middle", "late", "all three depths")
d2   <- lay[lay$lang == "en" & lay$block %in% keep, ]
d2$block <- factor(d2$block, levels = keep,
                   labels = c("early", "middle", "late", "all three\ndepths"))
d2$hi <- d2$auroc == max(d2$auroc)

p2 <- ggplot(d2, aes(x = block, y = auroc, fill = hi)) +
  geom_hline(yintercept = 0.5, colour = GREY, linewidth = 0.25, linetype = "33") +
  geom_col(width = 0.6) +
  geom_errorbar(aes(ymin = auroc - sd, ymax = auroc + sd),
                width = 0.14, colour = "#666666", linewidth = 0.32) +
  geom_text(aes(y = auroc + sd + 0.012, label = sprintf("%.3f", auroc)),
            size = 2.15, colour = INK) +
  scale_fill_manual(values = c(`FALSE` = BAR, `TRUE` = ACCENT), guide = "none") +
  coord_cartesian(ylim = c(0.45, 0.73)) +
  labs(x = NULL, y = "probe AUROC") +
  base_theme +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(colour = "#DDDDDD", linewidth = 0.25))
save_fig("fig_layers", p2, COL_W, 1.65)
