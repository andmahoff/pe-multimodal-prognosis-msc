# ---------------------------------------------------------
# fig_spesi_dca.R
# Net benefit: model ladder against sPESI-6
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  ggsave(p_pdf, plot, device = cairo_pdf,
         width = width, height = height,
         units = "in", bg = "white")
  message("saved: ", p_png)
  message("saved: ", p_pdf)
}

LAB <- c(death_30d = "30-Day Death",
         composite_30d = "Composite Outcome")

STRAT <- c("WMEAN3" = "Three-modality model",
           "WMEAN2" = "Two-modality model",
           "EHR" = "EHR modality alone",
           "sPESI" = "sPESI-6",
           "treat_all" = "Treat all",
           "treat_none" = "Treat none")

PAL <- c("Three-modality model" = "#0072B2",
         "Two-modality model" = "#56B4E9",
         "EHR modality alone" = "#009E73",
         "sPESI-6" = "#D55E00",
         "Treat all" = "#999999",
         "Treat none" = "grey45")

LTY <- c("Three-modality model" = "solid",
         "Two-modality model" = "solid",
         "EHR modality alone" = "solid",
         "sPESI-6" = "solid",
         "Treat all" = "dashed",
         "Treat none" = "dotted")

## ---- 2. one panel ---------------------------------------

dca_panel <- function(outcome) {
  d <- read.csv(file.path(
    data_dir, paste0("dca_", outcome, ".csv")),
    stringsAsFactors = FALSE)

  g <- d %>%
    pivot_longer(all_of(names(STRAT)),
                 names_to = "key", values_to = "nb") %>%
    mutate(strategy = factor(STRAT[key],
                             levels = unname(STRAT)))

  peak <- d %>%
    mutate(gain = WMEAN3 - sPESI) %>%
    slice_max(gain, n = 1)
  cat(sprintf("\n%s: peak gain over sPESI +%.4f at %.3f\n",
              outcome, peak$gain[1], peak$thr[1]))
  cat(sprintf("  sPESI turns negative at: %s\n",
              ifelse(any(d$sPESI < 0),
                     sprintf("%.3f",
                             min(d$thr[d$sPESI < 0])),
                     "never in range")))

  ymax <- max(g$nb[g$key != "treat_all"]) * 1.12

  ggplot(g, aes(x = thr, y = nb, colour = strategy,
                linetype = strategy)) +
    geom_hline(yintercept = 0, colour = "grey55",
               linewidth = 0.35) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = PAL, name = NULL) +
    scale_linetype_manual(values = LTY, name = NULL) +
    coord_cartesian(ylim = c(-0.02, ymax)) +
    scale_x_continuous(
      labels = function(x) sprintf("%.2f", x)) +
    labs(x = "Threshold probability",
         y = "Net benefit", title = LAB[outcome]) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(
            colour = "grey93", linewidth = 0.3),
          axis.title = element_text(size = 9),
          plot.title = element_text(
            size = 10, face = "bold", hjust = 0),
          plot.title.position = "plot",
          plot.margin = margin(4, 10, 2, 4))
}

## ---- 3. combine and save --------------------------------

p1 <- dca_panel("death_30d")
p2 <- dca_panel("composite_30d")

fD <- (p1 | p2) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = paste("Clinical Utility Against the",
                  "Guideline Risk Score"),
    theme = theme(
      plot.title = element_text(
        size = 12.5, face = "bold", hjust = 0.5,
        family = FONT, margin = margin(b = 12)))) &
  theme(legend.position = "bottom",
        legend.text = element_text(size = 9),
        legend.margin = margin(t = 2, b = 0))

print(fD)
save_fig(fD, "fig_spesi_dca", 10.4, 4.8)