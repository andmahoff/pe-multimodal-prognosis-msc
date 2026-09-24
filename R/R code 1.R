# ---------------------------------------------------------
# fig_roc_panel.R
# ROC curves by outcome, two-modality subsection
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(patchwork)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- TRUE   # set FALSE to skip the PDF entirely

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             cv_first = "CV Readmission")
OUT_ORD <- names(OUT_LAB)

MOD <- c("EHR (zero-shot)", "ECG", "WMEAN2", "sPESI-6")

PAL <- c("EHR (zero-shot)" = "#0072B2",
         "ECG"             = "#E69F00",
         "WMEAN2"          = "#D55E00",
         "sPESI-6"         = "#7F7F7F")

LTY <- c("EHR (zero-shot)" = "solid",
         "ECG"             = "22",
         "WMEAN2"          = "solid",
         "sPESI-6"         = "12")

## ---- 2. data --------------------------------------------

rc <- read.csv(file.path(data_dir, "roc_curves.csv"),
               stringsAsFactors = FALSE)
mt <- read.csv(file.path(data_dir, "roc_meta.csv"),
               stringsAsFactors = FALSE)
sp <- read.csv(file.path(data_dir,
                         "roc_spesi_point.csv"),
               stringsAsFactors = FALSE)

cat("\n--- AUROC ---\n")
print(mt, digits = 4)
cat("\nvertices per series:\n")
print(table(rc$outcome, rc$model))

keep <- intersect(OUT_ORD, unique(rc$outcome))

## ---- 3. one panel ---------------------------------------

roc_panel <- function(o, tag) {

  d <- rc %>%
    filter(outcome == o) %>%
    mutate(model = factor(model, levels = MOD))

  m <- mt %>%
    filter(outcome == o) %>%
    mutate(model = factor(model, levels = MOD)) %>%
    arrange(model)

  hdr <- sprintf("%s\nn = %s, %d events",
                 OUT_LAB[o],
                 format(max(m$n), big.mark = ","),
                 max(m$events))

  leg <- m %>%
    mutate(lab = sprintf("%s  %.4f", model, auc),
           yy = 0.30 - 0.055 * (row_number() - 1))

  pt <- sp %>% filter(outcome == o)

  ggplot(d, aes(x = fpr, y = tpr, colour = model,
                linetype = model)) +
    geom_abline(slope = 1, intercept = 0,
                colour = "grey75", linewidth = 0.35,
                linetype = "22") +
    geom_step(linewidth = 0.75) +
    geom_point(data = pt,
               aes(x = fpr, y = tpr),
               colour = "grey45", size = 2.1,
               inherit.aes = FALSE) +
    geom_segment(data = leg,
                 aes(x = 0.36, xend = 0.44,
                     y = yy, yend = yy,
                     colour = model,
                     linetype = model),
                 linewidth = 0.75,
                 inherit.aes = FALSE) +
    geom_text(data = leg,
              aes(x = 0.47, y = yy, label = lab),
              hjust = 0, size = 2.9, colour = "grey20",
              family = FONT, inherit.aes = FALSE) +
    scale_colour_manual(values = PAL, guide = "none",
                        drop = FALSE) +
    scale_linetype_manual(values = LTY,
                          guide = "none",
                          drop = FALSE) +
    scale_x_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = c(0.005, 0)) +
    scale_y_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = c(0.005, 0)) +
    coord_fixed() +
    labs(x = "1 \u2212 Specificity", y = "Sensitivity",
         title = hdr, tag = tag) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(
            colour = "grey93", linewidth = 0.3),
          axis.title = element_text(size = 9),
          plot.title = element_text(
            size = 9.5, face = "bold", hjust = 0.5,
            margin = margin(b = 6)),
          plot.tag = element_text(
            size = 12, face = "bold"),
          plot.tag.position = c(0.02, 0.99),
          plot.margin = margin(4, 8, 4, 4))
}

## ---- 4. combine and save --------------------------------

tags <- c("A", "B", "C", "D")
ps <- lapply(seq_along(keep), function(i)
  roc_panel(keep[i], tags[i]))

f <- wrap_plots(ps, ncol = 2) +
  plot_annotation(
    title = paste("Receiver Operating Characteristic",
                  "Curves by Outcome"),
    theme = theme(
      plot.title = element_text(
        size = 13, face = "bold", hjust = 0.5,
        family = FONT, margin = margin(b = 10))))

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig_roc_panel", 9.0, 9.4)))






























# ---------------------------------------------------------
# fig_roc3_panel.R
# ROC curves by outcome, three-modality subsection
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(patchwork)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- TRUE   # set FALSE to skip the PDF entirely

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             cv_first = "CV Readmission")
OUT_ORD <- names(OUT_LAB)

MOD <- c("EHR (zero-shot)", "CTPA report",
         "WMEAN2", "WMEAN3")

PAL <- c("EHR (zero-shot)" = "#0072B2",
         "CTPA report"     = "#009E73",
         "WMEAN2"          = "#7F7F7F",
         "WMEAN3"          = "#D55E00")

LTY <- c("EHR (zero-shot)" = "solid",
         "CTPA report"     = "22",
         "WMEAN2"          = "12",
         "WMEAN3"          = "solid")

## ---- 2. data --------------------------------------------

rc <- read.csv(file.path(data_dir, "roc3_curves.csv"),
               stringsAsFactors = FALSE)
mt <- read.csv(file.path(data_dir, "roc3_meta.csv"),
               stringsAsFactors = FALSE)

cat("\n--- AUROC ---\n")
print(mt, digits = 4)
cat("\nvertices per series:\n")
print(table(rc$outcome, rc$model))

keep <- intersect(OUT_ORD, unique(rc$outcome))

## ---- 3. one panel ---------------------------------------

roc_panel <- function(o, tag) {

  d <- rc %>%
    filter(outcome == o) %>%
    mutate(model = factor(model, levels = MOD))

  m <- mt %>%
    filter(outcome == o) %>%
    mutate(model = factor(model, levels = MOD)) %>%
    arrange(model)

  hdr <- sprintf("%s\nn = %s, %d events",
                 OUT_LAB[o],
                 format(max(m$n), big.mark = ","),
                 max(m$events))

  leg <- m %>%
    mutate(lab = sprintf("%s  %.4f", model, auc),
           yy = 0.30 - 0.055 * (row_number() - 1))

  ggplot(d, aes(x = fpr, y = tpr, colour = model,
                linetype = model)) +
    geom_abline(slope = 1, intercept = 0,
                colour = "grey75", linewidth = 0.35,
                linetype = "22") +
    geom_step(linewidth = 0.75) +
    geom_segment(data = leg,
                 aes(x = 0.34, xend = 0.42,
                     y = yy, yend = yy,
                     colour = model,
                     linetype = model),
                 linewidth = 0.75,
                 inherit.aes = FALSE) +
    geom_text(data = leg,
              aes(x = 0.45, y = yy, label = lab),
              hjust = 0, size = 2.9, colour = "grey20",
              family = FONT, inherit.aes = FALSE) +
    scale_colour_manual(values = PAL, guide = "none",
                        drop = FALSE) +
    scale_linetype_manual(values = LTY,
                          guide = "none",
                          drop = FALSE) +
    scale_x_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = c(0.005, 0)) +
    scale_y_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = c(0.005, 0)) +
    coord_fixed() +
    labs(x = "1 \u2212 Specificity", y = "Sensitivity",
         title = hdr, tag = tag) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(
            colour = "grey93", linewidth = 0.3),
          axis.title = element_text(size = 9),
          plot.title = element_text(
            size = 9.5, face = "bold", hjust = 0.5,
            margin = margin(b = 6)),
          plot.tag = element_text(
            size = 12, face = "bold"),
          plot.tag.position = c(0.02, 0.99),
          plot.margin = margin(4, 8, 4, 4))
}

## ---- 4. combine and save --------------------------------

tags <- c("A", "B", "C")
ps <- lapply(seq_along(keep), function(i)
  roc_panel(keep[i], tags[i]))

f <- wrap_plots(ps, ncol = 3) +
  plot_annotation(
    title = paste("Receiver Operating Characteristic",
                  "Curves on the CTPA Cohort"),
    theme = theme(
      plot.title = element_text(
        size = 13, face = "bold", hjust = 0.5,
        family = FONT, margin = margin(b = 10))))

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig_roc3_panel", 12.6, 4.8)))








































  # ---------------------------------------------------------
# fig_roc_common.R
# ROC curves for the four modalities, common cohort
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(patchwork)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- TRUE

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             cv_first = "CV Readmission")
OUT_ORD <- names(OUT_LAB)

MOD <- c("EHR", "ECG", "CTPA", "CXR")

PAL <- c(EHR = "#0072B2", ECG = "#E69F00",
         CTPA = "#009E73", CXR = "#CC79A7")

LTY <- c(EHR = "solid", ECG = "22",
         CTPA = "solid", CXR = "12")

# the CSVs hold the long modality names
SHORT <- c("Structured EHR"    = "EHR",
           "Electrocardiogram" = "ECG",
           "CTPA report"       = "CTPA",
           "Chest radiograph"  = "CXR")
short_name <- function(x) {
  ifelse(x %in% names(SHORT), SHORT[x], x)
}

## ---- 2. data --------------------------------------------

rc <- read.csv(file.path(
  data_dir, "roc_common_curves.csv"),
  stringsAsFactors = FALSE)
mt <- read.csv(file.path(
  data_dir, "roc_common_meta.csv"),
  stringsAsFactors = FALSE)
rc$model <- short_name(rc$model)
mt$model <- short_name(mt$model)

cat("\n--- AUROC on the common cohort ---\n")
print(mt, digits = 4)

chk <- mt %>%
  group_by(outcome) %>%
  summarise(n_distinct_n = n_distinct(n),
            .groups = "drop")
cat("\nsame patients across all modalities:\n")
print(as.data.frame(chk))
stopifnot(all(chk$n_distinct_n == 1))

keep <- intersect(OUT_ORD, unique(rc$outcome))

## ---- 3. one panel ---------------------------------------

roc_panel <- function(o, tag) {

  d <- rc %>%
    filter(outcome == o) %>%
    mutate(model = factor(model, levels = MOD))

  m <- mt %>%
    filter(outcome == o) %>%
    mutate(model = factor(model, levels = MOD)) %>%
    arrange(model)

  hdr <- sprintf("%s\nn = %s, %d events",
                 OUT_LAB[o],
                 format(m$n[1], big.mark = ","),
                 m$events[1])

  leg <- m %>%
    mutate(lab = sprintf("%s  %.4f", model, auc),
           yy = 0.30 - 0.055 * (row_number() - 1))

  ggplot(d, aes(x = fpr, y = tpr, colour = model,
                linetype = model)) +
    geom_abline(slope = 1, intercept = 0,
                colour = "grey75", linewidth = 0.35,
                linetype = "22") +
    geom_step(linewidth = 0.75) +
    geom_segment(data = leg,
                 aes(x = 0.30, xend = 0.38,
                     y = yy, yend = yy,
                     colour = model,
                     linetype = model),
                 linewidth = 0.75,
                 inherit.aes = FALSE) +
    geom_text(data = leg,
              aes(x = 0.41, y = yy, label = lab),
              hjust = 0, size = 2.9, colour = "grey20",
              family = FONT, inherit.aes = FALSE) +
    scale_colour_manual(values = PAL, guide = "none",
                        drop = FALSE) +
    scale_linetype_manual(values = LTY,
                          guide = "none",
                          drop = FALSE) +
    scale_x_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = c(0.005, 0)) +
    scale_y_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = c(0.005, 0)) +
    coord_fixed() +
    labs(x = "1 \u2212 Specificity", y = "Sensitivity",
         title = hdr, tag = tag) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(
            colour = "grey93", linewidth = 0.3),
          axis.title = element_text(size = 9),
          plot.title = element_text(
            size = 9.5, face = "bold", hjust = 0.5,
            margin = margin(b = 6)),
          plot.tag = element_text(
            size = 12, face = "bold"),
          plot.tag.position = c(0.02, 0.99),
          plot.margin = margin(4, 8, 4, 4))
}

## ---- 4. combine and save --------------------------------

tags <- c("A", "B", "C", "D")
ps <- lapply(seq_along(keep), function(i)
  roc_panel(keep[i], tags[i]))

f <- wrap_plots(ps, nrow = 1) +
  plot_annotation(
    title = paste("Discrimination of the Four Unimodal",
                  "Modalities on a Common Cohort"),
    theme = theme(
      plot.title = element_text(
        size = 13, face = "bold", hjust = 0.5,
        family = FONT, margin = margin(b = 10))))

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig_roc_common",
           4.3 * length(keep), 4.8)))














# ---------------------------------------------------------
# fig4_calibration.R
# Observed against predicted risk by decile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- TRUE

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             cv_first = "CV Readmission")

d <- read.csv(file.path(data_dir,
                        "calibration_deciles.csv"),
              stringsAsFactors = FALSE)
cat("\n--- deciles ---\n")
print(head(d, 12), digits = 3)

keep <- intersect(names(OUT_LAB), unique(d$outcome))

pd <- d %>%
  filter(outcome %in% keep) %>%
  mutate(outcome = factor(outcome, levels = keep,
                          labels = OUT_LAB[keep]))

f <- ggplot(pd, aes(x = pred, y = obs)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey60", linewidth = 0.4,
              linetype = "22") +
  geom_line(colour = "#0072B2", linewidth = 0.6,
            alpha = 0.7) +
  geom_point(colour = "#0072B2", size = 2.1) +
  facet_wrap(~ outcome, nrow = 1, scales = "free") +
  scale_x_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = 0.06)) +
  scale_y_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = 0.06)) +
  labs(x = "Mean Predicted Risk",
       y = "Observed Event Rate",
       title = paste("Calibration of the Fused Model",
                     "by Decile of Predicted Risk")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey93", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot")

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig4_calibration",
           3.4 * length(keep), 3.6)))























# ---------------------------------------------------------
# fig4_calibration.R
# Observed against predicted risk by decile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- FALSE

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white",
           family = "sans")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             cv_first = "CV Readmission")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "calibration_deciles.csv"),
              stringsAsFactors = FALSE)

cat("\n--- deciles ---\n")
print(head(d, 12), digits = 3)

keep <- intersect(names(OUT_LAB), unique(d$outcome))

pd <- d %>%
  filter(outcome %in% keep) %>%
  mutate(outcome = factor(outcome, levels = keep,
                          labels = OUT_LAB[keep]))

cat("\nobserved-to-predicted ratio by outcome:\n")
print(pd %>%
        group_by(outcome) %>%
        summarise(ratio = sum(obs * n) /
                    sum(pred * n),
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = pred, y = obs)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey60", linewidth = 0.4,
              linetype = "22") +
  geom_line(colour = "#0072B2", linewidth = 0.6,
            alpha = 0.7) +
  geom_point(colour = "#0072B2", size = 2.1) +
  facet_wrap(~ outcome, nrow = 1, scales = "free") +
  scale_x_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = 0.06)) +
  scale_y_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = 0.06)) +
  labs(x = "Mean Predicted Risk",
       y = "Observed Event Rate",
       title = paste("Calibration of the Fused Model",
                     "by Decile of Predicted Risk")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey93", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig4_calibration",
           3.4 * length(keep), 3.6)))
























# ---------------------------------------------------------
# fig11_subgroups.R
# Discrimination by age band and by sex
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(patchwork)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- FALSE

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white",
           family = "sans")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             cv_first = "CV Readmission")

PAL <- c("Fused model" = "#0072B2",
         "sPESI-6"     = "#D55E00")

## ---- 2. data --------------------------------------------

s <- read.csv(file.path(data_dir,
                        "subgroup_spesi.csv"),
              stringsAsFactors = FALSE)

cat("\ngrouping variables present:\n")
print(table(s$var))
cat("\nlevels by variable:\n")
print(unique(s[, c("var", "level")]))

reshape_block <- function(d) {
  bind_rows(
    d %>% transmute(outcome, level, n, ev,
                    src = "Fused model",
                    auc = model_auc, lo = m_lo,
                    hi = m_hi),
    d %>% transmute(outcome, level, n, ev,
                    src = "sPESI-6",
                    auc = spesi_auc, lo = s_lo,
                    hi = s_hi)) %>%
    mutate(outcome = factor(
      outcome, levels = names(OUT_LAB),
      labels = OUT_LAB),
      src = factor(src, levels = names(PAL)))
}

make_panel <- function(d, lv, xlab, ttl) {
  pd <- reshape_block(d) %>%
    mutate(level = factor(level, levels = lv))
  ggplot(pd, aes(x = level, y = auc, colour = src,
                 shape = src, group = src)) +
    geom_hline(yintercept = 0.5, colour = "grey55",
               linewidth = 0.35, linetype = "22") +
    geom_line(linewidth = 0.7, alpha = 0.55,
              position = position_dodge(width = 0.26)) +
    geom_errorbar(aes(ymin = lo, ymax = hi),
                  width = 0.13, linewidth = 0.4,
                  position = position_dodge(width = 0.26)) +
    geom_point(size = 2.5,
               position = position_dodge(width = 0.26)) +
    facet_wrap(~ outcome, nrow = 1) +
    scale_colour_manual(values = PAL, name = NULL) +
    scale_shape_manual(values = c(16, 15), name = NULL) +
    scale_y_continuous(
      breaks = seq(0.5, 1.0, by = 0.1),
      labels = function(x) sprintf("%.1f", x)) +
    labs(x = xlab, y = "AUROC", title = ttl) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(
            colour = "grey93", linewidth = 0.3),
          strip.text = element_text(size = 9,
                                    face = "bold"),
          axis.title = element_text(size = 9),
          plot.title = element_text(
            size = 10.5, face = "bold", hjust = 0),
          plot.title.position = "plot",
          legend.position = "bottom",
          legend.text = element_text(size = 9))
}

## ---- 3. build whichever blocks exist --------------------

AGE_VARS <- c("band", "age", "age_band")
SEX_VARS <- c("sex", "gender")

age <- s %>% filter(var %in% AGE_VARS)
sex <- s %>% filter(var %in% SEX_VARS)

cat("\nage rows:", nrow(age),
    " sex rows:", nrow(sex), "\n")

ps <- list()
if (nrow(age)) {
  lv <- c("<50", "50-64", "65-79", "80+")
  lv <- intersect(lv, unique(age$level))
  ps[[length(ps) + 1]] <- make_panel(
    age, lv, "Age Band", "A   By Age Band")
}
if (nrow(sex)) {
  lv <- unique(sex$level)
  ps[[length(ps) + 1]] <- make_panel(
    sex, lv, "Sex", "B   By Sex")
} else {
  message("no sex rows found - drawing age only")
}

f <- if (length(ps) == 2) {
  (ps[[1]] / ps[[2]]) +
    plot_layout(guides = "collect", heights = c(1, 0.9))
} else {
  ps[[1]]
}

f <- f + plot_annotation(
  title = paste("Discrimination by Age Band and Sex"),
  theme = theme(
    plot.title = element_text(
      size = 12.5, face = "bold", hjust = 0.5,
      family = FONT, margin = margin(b = 10)))) &
  theme(legend.position = "bottom")

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig11_subgroups", 11.0,
           ifelse(length(ps) == 2, 7.2, 4.2))))


















# ---------------------------------------------------------
# fig11_subgroups.R
# Discrimination by age band and by sex
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(patchwork)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

WRITE_PDF <- FALSE

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  message("saved: ", p_png)
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white",
           family = "sans")
    message("saved: ", p_pdf)
  }
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             cv_first = "CV Readmission")

PAL <- c("Fused model" = "#0072B2",
         "sPESI-6"     = "#D55E00")

## ---- 2. data --------------------------------------------

s <- read.csv(file.path(data_dir,
                        "subgroup_spesi.csv"),
              stringsAsFactors = FALSE)

cat("\ngrouping variables present:\n")
print(table(s$var))
cat("\nlevels by variable:\n")
print(unique(s[, c("var", "level")]))

reshape_block <- function(d) {
  bind_rows(
    d %>% transmute(outcome, level, n, ev,
                    src = "Fused model",
                    auc = model_auc, lo = m_lo,
                    hi = m_hi),
    d %>% transmute(outcome, level, n, ev,
                    src = "sPESI-6",
                    auc = spesi_auc, lo = s_lo,
                    hi = s_hi)) %>%
    mutate(outcome = factor(
      outcome, levels = names(OUT_LAB),
      labels = OUT_LAB),
      src = factor(src, levels = names(PAL)))
}

make_panel <- function(d, lv, xlab, ttl) {
  pd <- reshape_block(d) %>%
    mutate(level = factor(level, levels = lv))
  ggplot(pd, aes(x = level, y = auc, colour = src,
                 shape = src, group = src)) +
    geom_hline(yintercept = 0.5, colour = "grey55",
               linewidth = 0.35, linetype = "22") +
    geom_line(linewidth = 0.7, alpha = 0.55,
              position = position_dodge(width = 0.26)) +
    geom_errorbar(aes(ymin = lo, ymax = hi),
                  width = 0.13, linewidth = 0.4,
                  position = position_dodge(width = 0.26)) +
    geom_point(size = 2.5,
               position = position_dodge(width = 0.26)) +
    facet_wrap(~ outcome, nrow = 1) +
    scale_colour_manual(values = PAL, name = NULL) +
    scale_shape_manual(values = c(16, 15), name = NULL) +
    scale_y_continuous(
      breaks = seq(0.5, 1.0, by = 0.1),
      labels = function(x) sprintf("%.1f", x)) +
    labs(x = xlab, y = "AUROC", title = ttl) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(
            colour = "grey93", linewidth = 0.3),
          strip.text = element_text(size = 9,
                                    face = "bold"),
          axis.title = element_text(size = 9),
          plot.title = element_text(
            size = 10.5, face = "bold", hjust = 0),
          plot.title.position = "plot",
          legend.position = "bottom",
          legend.text = element_text(size = 9))
}

## ---- 3. build whichever blocks exist --------------------

AGE_VARS <- c("band", "age", "age_band")
SEX_VARS <- c("sex", "gender")

age <- s %>% filter(var %in% AGE_VARS)
sex <- s %>% filter(var %in% SEX_VARS)

cat("\nage rows:", nrow(age),
    " sex rows:", nrow(sex), "\n")

ps <- list()
if (nrow(age)) {
  lv <- c("<50", "50-64", "65-79", "80+")
  lv <- intersect(lv, unique(age$level))
  ps[[length(ps) + 1]] <- make_panel(
    age, lv, "Age Band", "A   By Age Band")
}
if (nrow(sex)) {
  lv <- unique(sex$level)
  ps[[length(ps) + 1]] <- make_panel(
    sex, lv, "Sex", "B   By Sex")
} else {
  message("no sex rows found - drawing age only")
}

f <- if (length(ps) == 2) {
  (ps[[1]] / ps[[2]]) +
    plot_layout(guides = "collect", heights = c(1, 0.9))
} else {
  ps[[1]]
}

f <- f + plot_annotation(
  title = paste("Discrimination by Age Band and Sex"),
  theme = theme(
    plot.title = element_text(
      size = 12.5, face = "bold", hjust = 0.5,
      family = FONT, margin = margin(b = 10)))) &
  theme(legend.position = "bottom")

cat("\nwriting files...\n")
print(system.time(
  save_fig(f, "fig11_subgroups", 11.0,
           ifelse(length(ps) == 2, 7.2, 4.2))))
