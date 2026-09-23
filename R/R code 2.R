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
# fig4_calibration.R
# Observed against predicted risk by decile, 2 x 2 layout
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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
  facet_wrap(~ outcome, ncol = 2, scales = "free") +
  scale_x_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = 0.07)) +
  scale_y_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = 0.07)) +
  labs(x = "Mean Predicted Risk",
       y = "Observed Event Rate",
       title = paste("Calibration of the Fused Model",
                     "by Decile of Predicted Risk")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey93", linewidth = 0.3),
        panel.spacing = unit(1.0, "lines"),
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
  save_fig(f, "fig4_calibration", 7.2, 6.8)))

















# ---------------------------------------------------------
# fig11_subgroups.R
# Discrimination by age band, and by sex, as two figures
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

## shared y-limits so the two figures are comparable
YLIM <- c(0.48, 0.98)

## ---- 2. data --------------------------------------------

s <- read.csv(file.path(data_dir,
                        "subgroup_spesi.csv"),
              stringsAsFactors = FALSE)

cat("\ngrouping variables present:\n")
print(table(s$var))

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

make_fig <- function(d, lv, xlab, ttl) {
  pd <- reshape_block(d) %>%
    mutate(level = factor(level, levels = lv))
  cat("\n---", ttl, "---\n")
  print(as.data.frame(
    pd %>% select(outcome, level, src, auc, lo, hi)),
    digits = 3)
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
      limits = YLIM,
      breaks = seq(0.5, 0.9, by = 0.1),
      labels = function(x) sprintf("%.1f", x)) +
    labs(x = xlab, y = "AUROC", title = ttl) +
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
          legend.position = "bottom",
          legend.text = element_text(size = 9.5),
          plot.margin = margin(6, 8, 4, 6))
}

## ---- 3. FIGURE 11a: age bands ---------------------------

AGE_VARS <- c("band", "age", "age_band")
age <- s %>% filter(var %in% AGE_VARS)
stopifnot(nrow(age) > 0)

lv_age <- intersect(c("<50", "50-64", "65-79", "80+"),
                    unique(age$level))
f_age <- make_fig(age, lv_age, "Age Band",
                  "Discrimination by Age Band")

cat("\nwriting figure 11a...\n")
print(system.time(
  save_fig(f_age, "fig11a_age_bands", 11.0, 4.4)))

## ---- 4. FIGURE 11b: sex ---------------------------------

SEX_VARS <- c("sex", "gender")
sex <- s %>% filter(var %in% SEX_VARS)

if (nrow(sex) == 0) {
  message("no sex rows found - figure 11b skipped")
} else {
  lv_sex <- unique(sex$level)
  f_sex <- make_fig(sex, lv_sex, "Sex",
                    "Discrimination by Sex")
  cat("\nwriting figure 11b...\n")
  print(system.time(
    save_fig(f_sex, "fig11b_sex", 11.0, 4.4)))
}





















# ---------------------------------------------------------
# fig21_timing.R
# Days to first composite event.
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

PAL <- c("All-cause in-hospital death" = "#0072B2",
         "Cardiovascular readmission"  = "#D55E00")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "time_to_event.csv"),
              stringsAsFactors = FALSE)

cat("\nevents:", nrow(d), "\n")
print(table(d$route))

cat("\ndays to event, overall:\n")
cat(sprintf("  mean %.2f  sd %.2f  median %.0f\n",
            mean(d$days), sd(d$days),
            median(d$days)))

cat("\ndays to event by route:\n")
print(d %>%
        group_by(route) %>%
        summarise(n = n(), mean = mean(days),
                  median = median(days),
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

d <- d %>%
  mutate(route = factor(route, levels = names(PAL)))

## ---- 3. FIGURE 21: overall distribution -----------------

f1 <- ggplot(d, aes(x = days)) +
  geom_histogram(binwidth = 1, boundary = -0.5,
                 fill = "#0072B2", colour = "white",
                 linewidth = 0.25, alpha = 0.9) +
  geom_vline(xintercept = median(d$days),
             colour = "grey35", linewidth = 0.5,
             linetype = "22") +
  annotate("text", x = median(d$days) + 0.7,
           y = Inf, vjust = 1.6, hjust = 0,
           label = sprintf("median %d days",
                           round(median(d$days))),
           size = 3.1, colour = "grey30",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0.01, 0)) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.10))) +
  labs(x = "Days From Admission to First Event",
       y = "Number of Events",
       title = paste("Time From Admission to First",
                     "Composite Event")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey93", linewidth = 0.3),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 21...\n")
print(system.time(
  save_fig(f1, "fig21_time_to_event", 7.6, 3.8)))

## ---- 4. FIGURE 22: split by route -----------------------

f2 <- ggplot(d, aes(x = days, fill = route)) +
  geom_histogram(binwidth = 1, boundary = -0.5,
                 colour = "white", linewidth = 0.25,
                 alpha = 0.9) +
  facet_wrap(~ route, ncol = 1) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0.01, 0)) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.10))) +
  labs(x = "Days From Admission to First Event",
       y = "Number of Events",
       title = paste("Deaths Occur Early and",
                     "Readmissions Late")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey93", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 22...\n")
print(system.time(
  save_fig(f2, "fig22_time_to_event_route", 7.6, 5.2)))


























# ---------------------------------------------------------
# fig23_twoway_disagree.R
# Fusion gain by inter-modality disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,
  "Composite (30-Day)", "Low",    205,  0.0004,
  "Composite (30-Day)", "Medium", 181,  0.0104,
  "Composite (30-Day)", "High",   161,  0.0230,
  "30-Day Death",       "Low",    160,  0.0011,
  "30-Day Death",       "Medium", 124,  0.0109,
  "30-Day Death",       "High",   114,  0.0191,
  "CV Readmission",     "Low",     82,  0.0009,
  "CV Readmission",     "Medium",  42,  0.0028,
  "CV Readmission",     "High",    47, -0.0035
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV))

cat("\n--- gains ---\n")
print(as.data.frame(pd), digits = 3)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey45",
             linewidth = 0.45) +
  geom_line(colour = "#0072B2", linewidth = 0.7,
            alpha = 0.6) +
  geom_point(colour = "#0072B2", size = 2.8) +
  geom_text(aes(label = sprintf("%+.4f", gain)),
            vjust = -1.1, size = 2.9,
            colour = "grey25", family = FONT) +
  geom_text(aes(label = sprintf("%d ev", events),
                y = -Inf), vjust = -0.8, size = 2.7,
            colour = "grey45", family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_y_continuous(
    limits = c(-0.010, 0.030),
    breaks = seq(-0.01, 0.03, by = 0.01),
    labels = function(x) sprintf("%+.2f", x)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain Over the Better Single Modality",
       title = paste("Fusion Helps Most Where the Modalities",
                     "Disagree")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey93", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 23...\n")
print(system.time(
  save_fig(f, "fig23_twoway_disagree", 9.0, 3.8)))

























# ---------------------------------------------------------
# fig23_twoway_disagree.R
# Fusion gain by inter-modality disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

## Okabe-Ito; sign is carried by colour and shape
PAL <- c("Fusion better" = "#0072B2",
         "Fusion worse"  = "#D55E00")
SHP <- c("Fusion better" = 16,
         "Fusion worse"  = 22)

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,
  "Composite (30-Day)", "Low",    205,  0.0004,
  "Composite (30-Day)", "Medium", 181,  0.0104,
  "Composite (30-Day)", "High",   161,  0.0230,
  "30-Day Death",       "Low",    160,  0.0011,
  "30-Day Death",       "Medium", 124,  0.0109,
  "30-Day Death",       "High",   114,  0.0191,
  "CV Readmission",     "Low",     82,  0.0009,
  "CV Readmission",     "Medium",  42,  0.0028,
  "CV Readmission",     "High",    47, -0.0035
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV),
         sign = factor(
           ifelse(gain >= 0, "Fusion better",
                  "Fusion worse"),
           levels = names(PAL)))

cat("\n--- gains ---\n")
print(as.data.frame(pd), digits = 3)
cat("\nmonotone increasing?\n")
print(pd %>%
        group_by(outcome) %>%
        summarise(mono = all(diff(gain) > 0),
                  .groups = "drop") %>%
        as.data.frame())

YLIM <- c(-0.009, 0.030)
Y_EV <- -0.0072   # fixed row for the event annotation

## direct label, placed once on the leftmost facet
lab_pt <- pd %>%
  filter(outcome == OUT_ORD[1], stratum == "High")

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_vline(xintercept = c(1, 2, 3),
             colour = "grey94", linewidth = 0.3) +
  geom_line(colour = "grey55", linewidth = 0.6) +
  geom_point(aes(colour = sign, shape = sign,
                 fill = sign),
             size = 2.9, stroke = 0.8) +
  geom_text(aes(label = sprintf("%+.4f", gain),
                colour = sign),
            vjust = -1.25, size = 2.9,
            family = FONT, show.legend = FALSE) +
  geom_text(aes(y = Y_EV,
                label = sprintf("%d events", events)),
            size = 2.6, colour = "grey45",
            family = FONT) +
  geom_text(data = lab_pt,
            aes(label = "gain over the better modality"),
            hjust = 1.06, vjust = 2.6, size = 2.9,
            colour = "grey35", family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_fill_manual(values = c(
    "Fusion better" = "#0072B2",
    "Fusion worse"  = "white"), guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_y_continuous(
    limits = YLIM,
    breaks = seq(-0.005, 0.030, by = 0.005),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain in AUROC Over the Better Single Modality",
       title = paste("Fusion Helps Most Where the Two",
                     "Modalities Disagree")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 23...\n")
print(system.time(
  save_fig(f, "fig23_twoway_disagree", 8.8, 3.9)))

























# ---------------------------------------------------------
# fig23_twoway_disagree.R
# Fusion gain by inter-modality disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

POS <- "#0072B2"   # Okabe-Ito blue
NEG <- "#D55E00"   # Okabe-Ito vermillion

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,
  "Composite (30-Day)", "Low",    205,  0.0004,
  "Composite (30-Day)", "Medium", 181,  0.0104,
  "Composite (30-Day)", "High",   161,  0.0230,
  "30-Day Death",       "Low",    160,  0.0011,
  "30-Day Death",       "Medium", 124,  0.0109,
  "30-Day Death",       "High",   114,  0.0191,
  "CV Readmission",     "Low",     82,  0.0009,
  "CV Readmission",     "Medium",  42,  0.0028,
  "CV Readmission",     "High",    47, -0.0035
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV))

cat("\n--- gains ---\n")
print(as.data.frame(pd), digits = 3)
cat("\nnegative gains:", sum(pd$gain < 0), "\n")
print(pd[pd$gain < 0, ])
cat("\nmonotone increasing?\n")
print(pd %>%
        group_by(outcome) %>%
        summarise(mono = all(diff(gain) > 0),
                  .groups = "drop") %>%
        as.data.frame())

## split explicitly, so shape and fill cannot misfire
pos <- pd %>% filter(gain >= 0)
neg <- pd %>% filter(gain < 0)

YLIM <- c(-0.009, 0.030)
Y_EV <- -0.0075

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain)) +
  geom_vline(xintercept = c(1, 2, 3),
             colour = "grey94", linewidth = 0.3) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(aes(group = outcome), colour = "grey55",
            linewidth = 0.6) +
  geom_point(data = pos, shape = 16, size = 2.9,
             colour = POS) +
  geom_point(data = neg, shape = 22, size = 2.7,
             colour = NEG, fill = "white",
             stroke = 0.9) +
  geom_text(data = pos,
            aes(label = sprintf("%+.4f", gain)),
            vjust = -1.3, size = 2.9, colour = POS,
            family = FONT) +
  geom_text(data = neg,
            aes(label = sprintf("%+.4f", gain)),
            vjust = 2.1, size = 2.9, colour = NEG,
            family = FONT) +
  geom_text(aes(y = Y_EV,
                label = sprintf("%d events", events)),
            size = 2.6, colour = "grey45",
            family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_y_continuous(
    limits = YLIM,
    breaks = seq(-0.005, 0.030, by = 0.005),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain in AUROC Over the Better Single Modality",
       title = paste("Fusion Helps Most Where the Two",
                     "Modalities Disagree")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 23...\n")
print(system.time(
  save_fig(f, "fig23_twoway_disagree", 8.8, 3.9)))





















# ---------------------------------------------------------
# fig23_twoway_disagree.R
# Fusion gain by inter-modality disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

POS <- "#0072B2"   # Okabe-Ito blue
NEG <- "#D55E00"   # Okabe-Ito vermillion

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,   ~hj,  ~vj,
  "Composite (30-Day)", "Low",    205,  0.0004,   1.18, -0.35,
  "Composite (30-Day)", "Medium", 181,  0.0104,   1.18, -0.35,
  "Composite (30-Day)", "High",   161,  0.0230,   1.12, -0.45,
  "30-Day Death",       "Low",    160,  0.0011,   1.18, -0.35,
  "30-Day Death",       "Medium", 124,  0.0109,   1.18, -0.35,
  "30-Day Death",       "High",   114,  0.0191,   1.12, -0.45,
  "CV Readmission",     "Low",     82,  0.0009,   1.18, -0.35,
  "CV Readmission",     "Medium",  42,  0.0028,   0.50, -1.35,
  "CV Readmission",     "High",    47, -0.0035,   1.18,  1.30
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV))

cat("\n--- gains ---\n")
print(as.data.frame(pd %>% select(-hj, -vj)),
      digits = 3)
cat("\nnegative gains:", sum(pd$gain < 0), "\n")
print(as.data.frame(pd[pd$gain < 0,
                       c("outcome", "stratum",
                         "gain")]))
cat("\nmonotone increasing?\n")
print(pd %>%
        group_by(outcome) %>%
        summarise(mono = all(diff(gain) > 0),
                  .groups = "drop") %>%
        as.data.frame())

pos <- pd %>% filter(gain >= 0)
neg <- pd %>% filter(gain < 0)

YLIM <- c(-0.009, 0.030)
Y_EV <- -0.0078

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain)) +
  geom_vline(xintercept = c(1, 2, 3),
             colour = "grey94", linewidth = 0.3) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(aes(group = outcome), colour = "grey55",
            linewidth = 0.6) +
  geom_point(data = pos, shape = 16, size = 2.9,
             colour = POS) +
  geom_point(data = neg, shape = 22, size = 2.7,
             colour = NEG, fill = "white",
             stroke = 0.9) +
  geom_text(data = pos,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = POS, family = FONT) +
  geom_text(data = neg,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = NEG, family = FONT) +
  geom_text(aes(y = Y_EV,
                label = sprintf("%d events", events)),
            size = 2.6, colour = "grey45",
            family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_x_discrete(expand = expansion(add = 0.62)) +
  scale_y_continuous(
    limits = YLIM,
    breaks = seq(-0.005, 0.030, by = 0.005),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain in AUROC Over the Better Single Modality",
       title = paste("Fusion Helps Most Where the Two",
                     "Modalities Disagree")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 23...\n")
print(system.time(
  save_fig(f, "fig23_twoway_disagree", 9.2, 3.9)))
























# ---------------------------------------------------------
# fig24_threeway_disagree.R
# Three-modality fusion gain by disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

POS <- "#0072B2"   # Okabe-Ito blue
NEG <- "#D55E00"   # Okabe-Ito vermillion

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,   ~hj,  ~vj,
  "Composite (30-Day)", "Low",    104,  0.0076,   1.18, -0.35,
  "Composite (30-Day)", "Medium",  61,  0.0404,   1.12, -0.45,
  "Composite (30-Day)", "High",    64,  0.0173,   1.18,  1.30,
  "30-Day Death",       "Low",     77,  0.0103,   1.18, -0.35,
  "30-Day Death",       "Medium",  39,  0.0044,   1.18,  1.30,
  "30-Day Death",       "High",    41,  0.0568,   1.12, -0.45
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV))

cat("\n--- gains ---\n")
print(as.data.frame(pd %>% select(-hj, -vj)),
      digits = 3)
cat("\nnegative gains:", sum(pd$gain < 0), "\n")
cat("\nmonotone increasing?\n")
print(pd %>%
        group_by(outcome) %>%
        summarise(mono = all(diff(gain) > 0),
                  .groups = "drop") %>%
        as.data.frame())

pos <- pd %>% filter(gain >= 0)
neg <- pd %>% filter(gain < 0)

YLIM <- c(-0.009, 0.062)
Y_EV <- -0.0072

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain)) +
  geom_vline(xintercept = c(1, 2, 3),
             colour = "grey94", linewidth = 0.3) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(aes(group = outcome), colour = "grey55",
            linewidth = 0.6) +
  geom_point(data = pos, shape = 16, size = 2.9,
             colour = POS) +
  geom_point(data = neg, shape = 22, size = 2.7,
             colour = NEG, fill = "white",
             stroke = 0.9) +
  geom_text(data = pos,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = POS, family = FONT) +
  geom_text(aes(y = Y_EV,
                label = sprintf("%d events", events)),
            size = 2.6, colour = "grey45",
            family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_x_discrete(expand = expansion(add = 0.62)) +
  scale_y_continuous(
    limits = YLIM,
    breaks = seq(0, 0.060, by = 0.010),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain in AUROC Over the Best Single Modality",
       title = paste("Three-Modality Fusion Gain by Degree",
                     "of Disagreement")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 24...\n")
print(system.time(
  save_fig(f, "fig24_threeway_disagree", 6.8, 3.9)))


















# ---------------------------------------------------------
# fig24_threeway_disagree.R
# Three-modality fusion gain by disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

POS <- "#0072B2"   # Okabe-Ito blue
NEG <- "#D55E00"   # Okabe-Ito vermillion

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,   ~hj,  ~vj,
  "Composite (30-Day)", "Low",    104,  0.0076,   1.18, -0.35,
  "Composite (30-Day)", "Medium",  61,  0.0404,   1.12, -0.50,
  "Composite (30-Day)", "High",    64,  0.0173,   1.18,  1.35,
  "30-Day Death",       "Low",     77,  0.0102,   1.18, -0.35,
  "30-Day Death",       "Medium",  39,  0.0044,   1.18,  1.35,
  "30-Day Death",       "High",    41,  0.0568,   1.12, -0.50,
  "CV Readmission",     "Low",     26,  0.0019,   1.18, -0.50,
  "CV Readmission",     "Medium",  27, -0.0082,   1.18, -0.50,
  "CV Readmission",     "High",    28, -0.0170,   1.18,  1.35
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV))

cat("\n--- gains ---\n")
print(as.data.frame(pd %>% select(-hj, -vj)),
      digits = 3)
cat("\nnegative gains:", sum(pd$gain < 0), "\n")
print(as.data.frame(pd[pd$gain < 0,
                       c("outcome", "stratum",
                         "gain")]))

pos <- pd %>% filter(gain >= 0)
neg <- pd %>% filter(gain < 0)

YLIM <- c(-0.027, 0.064)
Y_EV <- -0.0235

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain)) +
  geom_vline(xintercept = c(1, 2, 3),
             colour = "grey94", linewidth = 0.3) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(aes(group = outcome), colour = "grey55",
            linewidth = 0.6) +
  geom_point(data = pos, shape = 16, size = 2.9,
             colour = POS) +
  geom_point(data = neg, shape = 22, size = 2.7,
             colour = NEG, fill = "white",
             stroke = 0.9) +
  geom_text(data = pos,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = POS, family = FONT) +
  geom_text(data = neg,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = NEG, family = FONT) +
  geom_text(aes(y = Y_EV,
                label = sprintf("%d events", events)),
            size = 2.6, colour = "grey45",
            family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_x_discrete(expand = expansion(add = 0.62)) +
  scale_y_continuous(
    limits = YLIM,
    breaks = seq(-0.020, 0.060, by = 0.020),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain in AUROC Over the Best Single Modality",
       title = paste("Three-Modality Fusion Gain by Degree",
                     "of Disagreement")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 24...\n")
print(system.time(
  save_fig(f, "fig24_threeway_disagree", 9.2, 3.9)))



























# ---------------------------------------------------------
# fig24_threeway_disagree.R
# Three-modality fusion gain by disagreement tercile
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

POS <- "#0072B2"   # Okabe-Ito blue
NEG <- "#D55E00"   # Okabe-Ito vermillion

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome,             ~stratum, ~events, ~gain,   ~hj,  ~vj,
  "Composite (30-Day)", "Low",    104,  0.0076,   1.18, -0.35,
  "Composite (30-Day)", "Medium",  61,  0.0404,   1.12, -0.50,
  "Composite (30-Day)", "High",    64,  0.0173,   1.18,  1.35,
  "30-Day Death",       "Low",     77,  0.0102,   1.18, -0.35,
  "30-Day Death",       "Medium",  39,  0.0044,   1.18,  1.35,
  "30-Day Death",       "High",    41,  0.0568,   1.12, -0.50,
  "CV Readmission",     "Low",     26,  0.0019,   1.18,  1.45,
  "CV Readmission",     "Medium",  27, -0.0082,   0.50,  1.95,
  "CV Readmission",     "High",    28, -0.0170,   1.18,  1.45
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
LV <- c("Low", "Medium", "High")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         stratum = factor(stratum, levels = LV))

cat("\n--- gains ---\n")
print(as.data.frame(pd %>% select(-hj, -vj)),
      digits = 3)
cat("\nnegative gains:", sum(pd$gain < 0), "\n")
print(as.data.frame(pd[pd$gain < 0,
                       c("outcome", "stratum",
                         "gain")]))

pos <- pd %>% filter(gain >= 0)
neg <- pd %>% filter(gain < 0)

YLIM <- c(-0.029, 0.064)
Y_EV <- -0.0255

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = stratum, y = gain)) +
  geom_vline(xintercept = c(1, 2, 3),
             colour = "grey94", linewidth = 0.3) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(aes(group = outcome), colour = "grey55",
            linewidth = 0.6) +
  geom_point(data = pos, shape = 16, size = 2.9,
             colour = POS) +
  geom_point(data = neg, shape = 22, size = 2.7,
             colour = NEG, fill = "white",
             stroke = 0.9) +
  geom_text(data = pos,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = POS, family = FONT) +
  geom_text(data = neg,
            aes(label = sprintf("%+.4f", gain),
                hjust = hj, vjust = vj),
            size = 2.9, colour = NEG, family = FONT) +
  geom_text(aes(y = Y_EV,
                label = sprintf("%d events", events)),
            size = 2.6, colour = "grey45",
            family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_x_discrete(expand = expansion(add = 0.62)) +
  scale_y_continuous(
    limits = YLIM,
    breaks = seq(-0.020, 0.060, by = 0.020),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Inter-Modality Disagreement Tercile",
       y = "Gain in AUROC Over the Best Single Modality",
       title = paste("Three-Modality Fusion Gain by Degree",
                     "of Disagreement")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 24...\n")
print(system.time(
  save_fig(f, "fig24_threeway_disagree", 9.2, 4.0)))






















# ---------------------------------------------------------
# fig13_quartile_rates.R
# Event rate by quartile of each continuous feature
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

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

BLUE <- "#0072B2"

## which outcome to plot, and which features
OUTCOME <- "30-day death"
KEEP <- c("Urea nitrogen", "Respiratory rate",
          "Age", "Heart rate", "Anion gap",
          "White cell count")

d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"),
              stringsAsFactors = FALSE)

cat("\nfeatures available:\n")
print(sort(unique(d$feature)))

pd <- d %>%
  filter(outcome == OUTCOME, feature %in% KEEP) %>%
  mutate(feature = factor(feature, levels = KEEP),
         quartile = factor(quartile,
                           labels = c("Q1", "Q2",
                                      "Q3", "Q4")))

cat("\n--- plotted ---\n")
print(as.data.frame(pd), digits = 3)

base <- pd %>%
  summarise(r = weighted.mean(rate, n)) %>%
  pull(r)

f <- ggplot(pd, aes(x = quartile, y = rate)) +
  geom_hline(yintercept = base, colour = "grey60",
             linewidth = 0.4, linetype = "22") +
  geom_col(fill = BLUE, width = 0.62,
           alpha = 0.92) +
  geom_text(aes(label = sprintf("%.3f", rate)),
            vjust = -0.6, size = 2.8,
            colour = "grey25", family = FONT) +
  facet_wrap(~ feature, nrow = 2) +
  scale_y_continuous(
    limits = c(0, max(pd$rate) * 1.22),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  labs(x = "Quartile of Feature Value",
       y = "Observed Event Rate",
       title = paste("Event Rate Rises With Each",
                     "Leading Feature")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 13...\n")
print(system.time(
  save_fig(f, "fig13_quartile_rates", 8.4, 5.0)))


















# ---------------------------------------------------------
# fig14_rank_concordance.R
# Source vs target rank of mean absolute SHAP
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

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

BLUE <- "#0072B2"
ORANGE <- "#D55E00"
SHIFT_LAB <- 3        # label features shifting >= this

d <- read.csv(file.path(data_dir,
                        "shap_rank_concordance.csv"),
              stringsAsFactors = FALSE)

cat("\n--- concordance ---\n")
print(d %>%
        group_by(outcome) %>%
        summarise(rho = cor(rank_source, rank_target,
                            method = "spearman"),
                  max_shift = max(abs(shift)),
                  .groups = "drop") %>%
        as.data.frame(), digits = 4)

pd <- d %>%
  mutate(big = abs(shift) >= SHIFT_LAB)

cat("\nfeatures shifting", SHIFT_LAB, "+ ranks:\n")
print(pd %>% filter(big) %>%
        select(outcome, feature, rank_source,
               rank_target, shift) %>%
        as.data.frame(), digits = 3)

rho <- pd %>%
  group_by(outcome) %>%
  summarise(rho = cor(rank_source, rank_target,
                      method = "spearman"),
            .groups = "drop") %>%
  mutate(lab = sprintf("Spearman \u03c1 = %.3f", rho))

f <- ggplot(pd, aes(x = rank_source, y = rank_target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey65", linewidth = 0.4,
              linetype = "22") +
  geom_point(data = filter(pd, !big),
             shape = 16, size = 2.2, colour = BLUE,
             alpha = 0.85) +
  geom_point(data = filter(pd, big),
             shape = 22, size = 2.6, colour = ORANGE,
             fill = "white", stroke = 0.9) +
  geom_text(data = filter(pd, big),
            aes(label = feature),
            hjust = -0.15, size = 2.8,
            colour = ORANGE, family = FONT) +
  geom_text(data = rho,
            aes(x = 2, y = 27, label = lab),
            hjust = 0, size = 3.1, colour = "grey25",
            family = FONT, inherit.aes = FALSE) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_x_continuous(
    limits = c(0, 30), breaks = seq(0, 28, by = 7),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(0, 30), breaks = seq(0, 28, by = 7),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Rank at the Source Institution",
       y = "Rank at the Target Institution",
       title = paste("Feature Attribution Is Preserved",
                     "Across Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14...\n")
print(system.time(
  save_fig(f, "fig14_rank_concordance", 8.0, 4.4)))
















# ---------------------------------------------------------
# fig13_quartile_rates.R
# Event rate by quartile of each leading feature
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

QPAL <- c("Q1" = "#BDD7E7", "Q2" = "#6BAED6",
          "Q3" = "#2B8CBE", "Q4" = "#08519C")

OUTCOME <- "30-day death"
WANT <- c("Urea nitrogen", "Respiratory rate",
          "Age", "Heart rate", "Anion gap",
          "White cell count")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"),
              stringsAsFactors = FALSE)

cat("\nfeatures available:\n")
print(sort(unique(d$feature)))

KEEP <- intersect(WANT, unique(d$feature))
missing <- setdiff(WANT, KEEP)
if (length(missing)) {
  message("NOT FOUND, dropped: ",
          paste(missing, collapse = ", "))
}
stopifnot(length(KEEP) > 0)
cat("plotting", length(KEEP), "features\n")

pd <- d %>%
  filter(outcome == OUTCOME, feature %in% KEEP) %>%
  mutate(feature = factor(feature, levels = KEEP),
         quartile = factor(quartile,
                           levels = 1:4,
                           labels = c("Q1", "Q2",
                                      "Q3", "Q4")))

cat("\n--- plotted ---\n")
print(as.data.frame(pd), digits = 3)

base <- weighted.mean(pd$rate, pd$n)
cat("\ncohort event rate:", sprintf("%.4f", base), "\n")

hatch <- do.call(rbind, lapply(seq_len(nrow(pd)),
  function(i) {
    r <- pd[i, ]
    k <- as.integer(r$quartile)
    if (k < 2) return(NULL)
    ys <- seq(0, r$rate,
              length.out = c(1, 3, 5, 8)[k] + 1)
    ys <- ys[-1]
    data.frame(feature = r$feature,
               quartile = r$quartile,
               x = as.integer(r$quartile),
               y = ys)
  }))

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = quartile, y = rate)) +
  geom_hline(yintercept = base, colour = "grey55",
             linewidth = 0.45, linetype = "22") +
  geom_col(aes(fill = quartile), width = 0.68,
           colour = "grey30", linewidth = 0.25) +
  geom_segment(data = hatch,
               aes(x = x - 0.34, xend = x + 0.34,
                   y = y, yend = y),
               colour = "white", linewidth = 0.3,
               inherit.aes = FALSE) +
  geom_text(aes(label = sprintf("%.3f", rate)),
            vjust = -0.65, size = 2.8,
            colour = "grey20", family = FONT) +
  facet_wrap(~ feature, nrow = 1) +
  scale_fill_manual(values = QPAL, guide = "none") +
  scale_y_continuous(
    limits = c(0, max(pd$rate) * 1.20),
    breaks = seq(0, 0.30, by = 0.05),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  labs(x = "Quartile of Feature Value",
       y = "Observed Event Rate",
       title = paste("Event Rate Rises Across Quartiles",
                     "of Each Leading Feature")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing = unit(0.9, "lines"),
        strip.text = element_text(
          size = 9.5, face = "bold",
          margin = margin(b = 5)),
        strip.placement = "outside",
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 13...\n")
print(system.time(
  save_fig(f, "fig13_quartile_rates",
           2.0 * length(KEEP) + 0.9, 4.0)))












d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"))
sort(unique(d$feature))






# ---------------------------------------------------------
# fig13_quartile_rates.R
# Event rate by quartile of each leading feature
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

## stepped lightness; darker with quartile
QPAL <- c("Q1" = "#BDD7E7", "Q2" = "#6BAED6",
          "Q3" = "#2B8CBE", "Q4" = "#08519C")

## rule spacing in data units: tighter with quartile
SPACING <- c(Inf, 0.030, 0.018, 0.010)

OUTCOME <- "30-day death"
WANT <- c("Urea nitrogen", "Respiratory rate",
          "Age", "Heart rate", "Anion gap",
          "White cell count")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"),
              stringsAsFactors = FALSE)

cat("\nfeatures available:\n")
print(sort(unique(d$feature)))

KEEP <- intersect(WANT, unique(d$feature))
missing <- setdiff(WANT, KEEP)
if (length(missing)) {
  message("NOT FOUND, dropped: ",
          paste(missing, collapse = ", "))
}
stopifnot(length(KEEP) > 0)
cat("plotting", length(KEEP), "features\n")

pd <- d %>%
  filter(outcome == OUTCOME, feature %in% KEEP) %>%
  mutate(feature = factor(feature, levels = KEEP),
         quartile = factor(quartile,
                           levels = 1:4,
                           labels = c("Q1", "Q2",
                                      "Q3", "Q4")))

cat("\n--- plotted ---\n")
print(as.data.frame(pd), digits = 3)

base <- weighted.mean(pd$rate, pd$n)
cat("\ncohort event rate:", sprintf("%.4f", base), "\n")

## ---- 3. hatch overlay -----------------------------------

hatch <- do.call(rbind, lapply(seq_len(nrow(pd)),
  function(i) {
    r <- pd[i, ]
    k <- as.integer(r$quartile)
    sp <- SPACING[k]
    if (!is.finite(sp)) return(NULL)
    ys <- seq(sp, r$rate - sp * 0.4, by = sp)
    if (length(ys) < 1) return(NULL)
    data.frame(feature = r$feature,
               quartile = r$quartile,
               x = as.integer(r$quartile),
               y = ys)
  }))

cat("\nhatch rules drawn:",
    ifelse(is.null(hatch), 0, nrow(hatch)), "\n")

## ---- 4. plot and save -----------------------------------

f <- ggplot(pd, aes(x = quartile, y = rate)) +
  geom_hline(yintercept = base, colour = "grey55",
             linewidth = 0.45, linetype = "22") +
  geom_col(aes(fill = quartile), width = 0.68,
           colour = "grey30", linewidth = 0.25)

if (!is.null(hatch)) {
  f <- f + geom_segment(
    data = hatch,
    aes(x = x - 0.34, xend = x + 0.34,
        y = y, yend = y),
    colour = "white", linewidth = 0.3,
    inherit.aes = FALSE)
}

f <- f +
  geom_text(aes(label = sprintf("%.3f", rate)),
            vjust = -0.65, size = 2.8,
            colour = "grey20", family = FONT) +
  facet_wrap(~ feature, nrow = 1) +
  scale_fill_manual(values = QPAL, guide = "none") +
  scale_y_continuous(
    limits = c(0, max(pd$rate) * 1.20),
    breaks = seq(0, 0.30, by = 0.05),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  labs(x = "Quartile of Feature Value",
       y = "Observed Event Rate",
       title = paste("Event Rate Rises Across Quartiles",
                     "of Each Leading Feature")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing = unit(0.9, "lines"),
        strip.text = element_text(
          size = 9.5, face = "bold",
          margin = margin(b = 5)),
        strip.placement = "outside",
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 13...\n")
print(system.time(
  save_fig(f, "fig13_quartile_rates",
           2.0 * length(KEEP) + 0.9, 4.0)))























# ---------------------------------------------------------
# fig13_quartile_rates.R
# Event rate by quartile of each leading feature
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

## stepped lightness; darker with quartile
QPAL <- c("Q1" = "#BDD7E7", "Q2" = "#6BAED6",
          "Q3" = "#2B8CBE", "Q4" = "#08519C")

## rule spacing in data units: tighter with quartile
SPACING <- c(Inf, 0.030, 0.020, 0.014)

NCOL <- 3          # 3 x 2 layout for six features

OUTCOME <- "30-day death"
WANT <- c("Urea nitrogen", "rr", "Age", "hr",
          "aniongap", "White cell count")

## the export left some feature names untranslated
RENAME <- c("rr" = "Respiratory rate",
            "hr" = "Heart rate",
            "aniongap" = "Anion gap",
            "sbp" = "Systolic BP",
            "dbp" = "Diastolic BP",
            "plt" = "Platelets",
            "rbc" = "Red cell count",
            "temp" = "Temperature",
            "bicarb" = "Bicarbonate")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"),
              stringsAsFactors = FALSE)

d$feature <- ifelse(d$feature %in% names(RENAME),
                    RENAME[d$feature], d$feature)

cat("\nfeatures available:\n")
print(sort(unique(d$feature)))

WANT_DISP <- unname(ifelse(WANT %in% names(RENAME),
                           RENAME[WANT], WANT))
KEEP <- intersect(WANT_DISP, unique(d$feature))
missing <- setdiff(WANT_DISP, KEEP)
if (length(missing)) {
  message("NOT FOUND, dropped: ",
          paste(missing, collapse = ", "))
}
stopifnot(length(KEEP) > 0)
cat("plotting", length(KEEP), "features\n")

pd <- d %>%
  filter(outcome == OUTCOME, feature %in% KEEP) %>%
  mutate(feature = factor(feature, levels = KEEP),
         quartile = factor(quartile,
                           levels = 1:4,
                           labels = c("Q1", "Q2",
                                      "Q3", "Q4")))

cat("\n--- plotted ---\n")
print(as.data.frame(pd), digits = 3)

cat("\nmonotone across quartiles?\n")
print(pd %>%
        arrange(feature, quartile) %>%
        group_by(feature) %>%
        summarise(mono = all(diff(rate) > 0),
                  .groups = "drop") %>%
        as.data.frame())

base <- weighted.mean(pd$rate, pd$n)
cat("\ncohort event rate:", sprintf("%.4f", base), "\n")

## ---- 3. hatch overlay -----------------------------------

hatch <- do.call(rbind, lapply(seq_len(nrow(pd)),
  function(i) {
    r <- pd[i, ]
    k <- as.integer(r$quartile)
    sp <- SPACING[k]
    if (!is.finite(sp)) return(NULL)
    ys <- seq(sp, r$rate - sp * 0.4, by = sp)
    if (length(ys) < 1) return(NULL)
    data.frame(feature = r$feature,
               quartile = r$quartile,
               x = as.integer(r$quartile),
               y = ys)
  }))

cat("\nhatch rules drawn:",
    ifelse(is.null(hatch), 0, nrow(hatch)), "\n")

## ---- 4. plot and save -----------------------------------

f <- ggplot(pd, aes(x = quartile, y = rate)) +
  geom_hline(yintercept = base, colour = "grey55",
             linewidth = 0.45, linetype = "22") +
  geom_col(aes(fill = quartile), width = 0.68,
           colour = "grey30", linewidth = 0.25)

if (!is.null(hatch)) {
  f <- f + geom_segment(
    data = hatch,
    aes(x = x - 0.34, xend = x + 0.34,
        y = y, yend = y),
    colour = "white", linewidth = 0.3,
    inherit.aes = FALSE)
}

f <- f +
  geom_text(aes(label = sprintf("%.3f", rate)),
            vjust = -0.65, size = 2.7,
            colour = "grey20", family = FONT) +
  facet_wrap(~ feature, ncol = NCOL) +
  scale_fill_manual(values = QPAL, guide = "none") +
  scale_y_continuous(
    limits = c(0, max(pd$rate) * 1.20),
    breaks = seq(0, 0.30, by = 0.05),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  labs(x = "Quartile of Feature Value",
       y = "Observed Event Rate",
       title = paste("Event Rate Rises Across Quartiles",
                     "of Each Leading Feature")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing = unit(0.9, "lines"),
        strip.text = element_text(
          size = 9.5, face = "bold",
          margin = margin(b = 5)),
        strip.placement = "outside",
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

nrows <- ceiling(length(KEEP) / NCOL)
cat("\nwriting figure 13...\n")
print(system.time(
  save_fig(f, "fig13_quartile_rates",
           2.5 * min(NCOL, length(KEEP)) + 0.9,
           2.6 * nrows + 0.9)))


























           # ---------------------------------------------------------
# fig13_quartile_rates.R
# Event rate by quartile of each leading feature
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

QPAL <- c("Q1" = "#BDD7E7", "Q2" = "#6BAED6",
          "Q3" = "#2B8CBE", "Q4" = "#08519C")

SPACING <- c(Inf, 0.030, 0.020, 0.014)

NCOL <- 3

OUTCOME <- "30-day death"
WANT <- c("Urea nitrogen", "rr", "Age", "hr",
          "aniongap", "White cell count")

RENAME <- c("rr" = "Respiratory rate",
            "hr" = "Heart rate",
            "aniongap" = "Anion gap",
            "sbp" = "Systolic BP",
            "dbp" = "Diastolic BP",
            "plt" = "Platelets",
            "rbc" = "Red cell count",
            "temp" = "Temperature",
            "bicarb" = "Bicarbonate")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"),
              stringsAsFactors = FALSE)

d$feature <- ifelse(d$feature %in% names(RENAME),
                    RENAME[d$feature], d$feature)

cat("\nfeatures available:\n")
print(sort(unique(d$feature)))

WANT_DISP <- unname(ifelse(WANT %in% names(RENAME),
                           RENAME[WANT], WANT))
KEEP <- intersect(WANT_DISP, unique(d$feature))
missing <- setdiff(WANT_DISP, KEEP)
if (length(missing)) {
  message("NOT FOUND, dropped: ",
          paste(missing, collapse = ", "))
}
stopifnot(length(KEEP) > 0)
cat("plotting", length(KEEP), "features\n")

pd <- d %>%
  filter(outcome == OUTCOME, feature %in% KEEP) %>%
  mutate(feature = factor(feature, levels = KEEP),
         quartile = factor(quartile,
                           levels = 1:4,
                           labels = c("Q1", "Q2",
                                      "Q3", "Q4")))

cat("\n--- plotted ---\n")
print(as.data.frame(pd), digits = 3)

cat("\nmonotone across quartiles?\n")
print(pd %>%
        arrange(feature, quartile) %>%
        group_by(feature) %>%
        summarise(mono = all(diff(rate) > 0),
                  .groups = "drop") %>%
        as.data.frame())

base <- weighted.mean(pd$rate, pd$n)
cat("\ncohort event rate:", sprintf("%.4f", base), "\n")

## ---- 3. hatch overlay -----------------------------------

hatch <- do.call(rbind, lapply(seq_len(nrow(pd)),
  function(i) {
    r <- pd[i, ]
    k <- as.integer(r$quartile)
    sp <- SPACING[k]
    if (!is.finite(sp)) return(NULL)
    ys <- seq(sp, r$rate - sp * 0.4, by = sp)
    if (length(ys) < 1) return(NULL)
    data.frame(feature = r$feature,
               quartile = r$quartile,
               x = as.integer(r$quartile),
               y = ys)
  }))

cat("\nhatch rules drawn:",
    ifelse(is.null(hatch), 0, nrow(hatch)), "\n")

## ---- 4. plot and save -----------------------------------

f <- ggplot(pd, aes(x = quartile, y = rate)) +
  geom_hline(yintercept = base, colour = "grey55",
             linewidth = 0.45, linetype = "22") +
  geom_col(aes(fill = quartile), width = 0.68,
           colour = "grey30", linewidth = 0.25)

if (!is.null(hatch)) {
  f <- f + geom_segment(
    data = hatch,
    aes(x = x - 0.34, xend = x + 0.34,
        y = y, yend = y),
    colour = "white", linewidth = 0.3,
    inherit.aes = FALSE)
}

f <- f +
  geom_text(aes(label = sprintf("%.3f", rate)),
            vjust = -0.65, size = 2.7,
            colour = "grey20", family = FONT) +
  facet_wrap(~ feature, ncol = NCOL,
             strip.position = "bottom") +
  scale_fill_manual(values = QPAL, guide = "none") +
  scale_y_continuous(
    limits = c(0, max(pd$rate) * 1.20),
    breaks = seq(0, 0.30, by = 0.05),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  labs(x = "Quartile of Feature Value",
       y = "Observed Event Rate",
       title = paste("Event Rate Rises Across Quartiles",
                     "of Each Leading Feature")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.x = unit(0.9, "lines"),
        panel.spacing.y = unit(1.4, "lines"),
        strip.placement = "outside",
        strip.background = element_blank(),
        strip.text = element_text(
          size = 9.5, face = "bold",
          margin = margin(t = 4, b = 2)),
        axis.text = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 8)),
        axis.title.y = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

nrows <- ceiling(length(KEEP) / NCOL)
cat("\nwriting figure 13...\n")
print(system.time(
  save_fig(f, "fig13_quartile_rates",
           2.5 * min(NCOL, length(KEEP)) + 0.9,
           2.8 * nrows + 1.0)))






















# ---------------------------------------------------------
# fig8_fusion_architectures.R
# Every three-modality combination rule vs the weighted average
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

POS <- "#0072B2"   # Okabe-Ito blue
NEG <- "#D55E00"   # Okabe-Ito vermillion

## ---- 2. data --------------------------------------------

d <- tribble(
  ~family, ~rule, ~outcome, ~diff, ~lo, ~hi,

  "Fixed rule", "Equal-weight average (MEAN3)",
  "Composite (30-Day)", -0.0066, NA, NA,
  "Fixed rule", "Equal-weight average (MEAN3)",
  "30-Day Death", -0.0083, NA, NA,

  "Meta-learner", "Linear stack",
  "Composite (30-Day)", -0.0021, -0.0039, -0.0003,
  "Meta-learner", "Linear stack",
  "30-Day Death", -0.0014, -0.0039, 0.0009,

  "Meta-learner", "Random forest",
  "Composite (30-Day)", -0.0050, -0.0145, 0.0047,
  "Meta-learner", "Random forest",
  "30-Day Death", -0.0008, -0.0141, 0.0124,

  "Meta-learner", "Gradient boosting",
  "Composite (30-Day)", -0.0092, -0.0201, 0.0020,
  "Meta-learner", "Gradient boosting",
  "30-Day Death", -0.0109, -0.0229, 0.0012,

  "Meta-learner", "Neural network, 8 units",
  "Composite (30-Day)", -0.0079, -0.0117, -0.0042,
  "Meta-learner", "Neural network, 8 units",
  "30-Day Death", -0.0072, -0.0118, -0.0027,

  "Meta-learner", "Neural network, 16-8 units",
  "Composite (30-Day)", -0.0055, -0.0086, -0.0025,
  "Meta-learner", "Neural network, 16-8 units",
  "30-Day Death", -0.0041, -0.0079, -0.0004,

  "Joint representation", "Joint branch model",
  "Composite (30-Day)", -0.0250, -0.0467, -0.0051,
  "Joint representation", "Joint branch model",
  "30-Day Death", 0.0051, -0.0224, 0.0309
)

FAM <- c("Fixed rule", "Meta-learner",
         "Joint representation")
OUT_ORD <- c("Composite (30-Day)", "30-Day Death")

## order rules by composite difference within family
ord <- d %>%
  filter(outcome == OUT_ORD[1]) %>%
  mutate(family = factor(family, levels = FAM)) %>%
  arrange(family, diff) %>%
  pull(rule)

pd <- d %>%
  mutate(family = factor(family, levels = FAM),
         rule = factor(rule, levels = rev(ord)),
         outcome = factor(outcome, levels = OUT_ORD),
         sig = !is.na(lo) & (lo > 0 | hi < 0))

cat("\n--- differences from WMEAN3 ---\n")
print(as.data.frame(pd %>% select(-family)),
      digits = 3)
cat("\nsignificant:", sum(pd$sig), "of", nrow(pd), "\n")
cat("positive:", sum(pd$diff > 0), "\n")

sigd <- pd %>% filter(sig)
nsd <- pd %>% filter(!sig)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = diff, y = rule)) +
  geom_vline(xintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 height = 0, linewidth = 0.5,
                 colour = "grey55", na.rm = TRUE) +
  geom_point(data = sigd, shape = 16, size = 2.5,
             colour = NEG) +
  geom_point(data = nsd, shape = 21, size = 2.4,
             colour = POS, fill = "white",
             stroke = 0.8) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_x_continuous(
    limits = c(-0.050, 0.035),
    breaks = seq(-0.04, 0.03, by = 0.02),
    labels = function(x) sprintf("%+.2f", x)) +
  labs(x = paste("Change in AUROC Relative to the",
                 "Weighted Average"),
       y = NULL,
       title = paste("No Learned Combination Rule",
                     "Improves on a Tuned Weight")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3,
          linetype = "dotted"),
        panel.grid.major.x = element_line(
          colour = "grey93", linewidth = 0.3),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text.y = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 8)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 10, 4, 6))

cat("\nwriting figure 8...\n")
print(system.time(
  save_fig(f, "fig8_fusion_architectures", 9.0, 4.2)))






























# ---------------------------------------------------------
# fig8_fusion_architectures.R
# Every learned combination rule vs the weighted average
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

NS <- "#0072B2"   # Okabe-Ito blue: interval spans zero
SIG <- "#D55E00"  # Okabe-Ito vermillion: excludes zero

## ---- 2. data --------------------------------------------

d <- tribble(
  ~family, ~rule, ~outcome, ~diff, ~lo, ~hi,

  "Fixed rule", "Equal-weight average (MEAN3)",
  "Composite (30-Day)", -0.0066, NA, NA,
  "Fixed rule", "Equal-weight average (MEAN3)",
  "30-Day Death", -0.0083, NA, NA,

  "Meta-learner", "Linear stack",
  "Composite (30-Day)", -0.0021, -0.0039, -0.0003,
  "Meta-learner", "Linear stack",
  "30-Day Death", -0.0014, -0.0039, 0.0009,

  "Meta-learner", "Random forest",
  "Composite (30-Day)", -0.0050, -0.0145, 0.0047,
  "Meta-learner", "Random forest",
  "30-Day Death", -0.0008, -0.0141, 0.0124,

  "Meta-learner", "Gradient boosting",
  "Composite (30-Day)", -0.0092, -0.0201, 0.0020,
  "Meta-learner", "Gradient boosting",
  "30-Day Death", -0.0109, -0.0229, 0.0012,

  "Meta-learner", "Neural network, 8 units",
  "Composite (30-Day)", -0.0079, -0.0117, -0.0042,
  "Meta-learner", "Neural network, 8 units",
  "30-Day Death", -0.0072, -0.0118, -0.0027,

  "Meta-learner", "Neural network, 16-8 units",
  "Composite (30-Day)", -0.0055, -0.0086, -0.0025,
  "Meta-learner", "Neural network, 16-8 units",
  "30-Day Death", -0.0041, -0.0079, -0.0004,

  "Joint model", "Joint branch model",
  "Composite (30-Day)", -0.0250, -0.0467, -0.0051,
  "Joint model", "Joint branch model",
  "30-Day Death", 0.0051, -0.0224, 0.0309
)

FAM <- c("Fixed rule", "Meta-learner", "Joint model")
OUT_ORD <- c("Composite (30-Day)", "30-Day Death")

ord <- d %>%
  filter(outcome == OUT_ORD[1]) %>%
  mutate(family = factor(family, levels = FAM)) %>%
  arrange(family, desc(diff)) %>%
  pull(rule)

pd <- d %>%
  mutate(family = factor(family, levels = FAM),
         rule = factor(rule, levels = rev(ord)),
         outcome = factor(outcome, levels = OUT_ORD),
         sig = !is.na(lo) & (lo > 0 | hi < 0))

cat("\n--- differences from WMEAN3 ---\n")
print(as.data.frame(pd), digits = 3)
cat("\nsignificant:", sum(pd$sig), "of", nrow(pd),
    " positive:", sum(pd$diff > 0), "\n")

XLIM <- c(-0.052, 0.046)
X_LAB <- 0.044          # fixed column for value labels

key <- tibble(
  outcome = factor(OUT_ORD[1], levels = OUT_ORD),
  family = factor("Fixed rule", levels = FAM),
  rule = factor(ord[1], levels = rev(ord)))

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = diff, y = rule)) +
  geom_vline(xintercept = 0, colour = "grey35",
             linewidth = 0.5) +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 height = 0, linewidth = 0.5,
                 colour = "grey60", na.rm = TRUE) +
  geom_point(aes(colour = sig, fill = sig,
                 shape = sig), size = 2.6,
             stroke = 0.9) +
  geom_text(aes(x = X_LAB,
                label = sprintf("%+.4f", diff),
                colour = sig),
            hjust = 1, size = 2.8, family = FONT,
            show.legend = FALSE) +
  geom_text(data = key,
            aes(x = -0.050, y = rule,
                label = "filled = interval excludes zero"),
            hjust = 0, vjust = -1.6, size = 2.7,
            colour = "grey35", family = FONT,
            inherit.aes = FALSE) +
  facet_grid(family ~ outcome, scales = "free_y",
             space = "free_y") +
  scale_colour_manual(
    values = c("TRUE" = SIG, "FALSE" = NS),
    guide = "none") +
  scale_fill_manual(
    values = c("TRUE" = SIG, "FALSE" = "white"),
    guide = "none") +
  scale_shape_manual(
    values = c("TRUE" = 21, "FALSE" = 21),
    guide = "none") +
  scale_x_continuous(
    limits = XLIM,
    breaks = seq(-0.04, 0.02, by = 0.02),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  labs(x = paste("Change in AUROC Relative to the",
                 "Weighted Average"),
       y = NULL,
       title = paste("No Learned Combination Rule",
                     "Improves on a Tuned Weight")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey95", linewidth = 0.3,
          linetype = "dotted"),
        panel.grid.major.x = element_line(
          colour = "grey93", linewidth = 0.3),
        panel.spacing.y = unit(0.35, "lines"),
        panel.spacing.x = unit(0.7, "lines"),
        strip.text.x = element_text(size = 9.5,
                                    face = "bold"),
        strip.text.y = element_text(
          size = 8.5, face = "bold", angle = 0,
          hjust = 0, colour = "grey30"),
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 7)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 8...\n")
print(system.time(
  save_fig(f, "fig8_fusion_architectures", 9.4, 4.4)))

























  # ---------------------------------------------------------
# fig7_transfer_cost.R
# Feature-restriction cost vs institution cost
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

PAL <- c("Restricting the feature set" = "#56B4E9",
         "Training at another institution" = "#0072B2")

d <- tribble(
  ~outcome,              ~component, ~cost,
  "30-Day Death",        "feat", 0.0202,
  "30-Day Death",        "inst", 0.0755,
  "In-Hospital Death",   "feat", 0.0297,
  "In-Hospital Death",   "inst", 0.0945,
  "Composite (30-Day)",  "feat", 0.0178,
  "Composite (30-Day)",  "inst", 0.0471
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "In-Hospital Death")

pd <- d %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD),
         component = factor(
           ifelse(component == "feat",
                  "Restricting the feature set",
                  "Training at another institution"),
           levels = names(PAL))) %>%
  arrange(outcome, component) %>%
  group_by(outcome) %>%
  mutate(hi = cumsum(cost), lo = hi - cost,
         mid = (lo + hi) / 2,
         total = sum(cost),
         ratio = cost[2] / cost[1]) %>%
  ungroup() %>%
  mutate(yy = as.numeric(outcome))

cat("\n--- cost decomposition ---\n")
print(as.data.frame(pd %>%
  select(outcome, component, cost, total)), digits = 3)
cat("\ninstitution-to-feature ratio:\n")
print(pd %>% distinct(outcome, ratio) %>%
        as.data.frame(), digits = 3)

HH <- 0.30

hatch <- do.call(rbind, lapply(
  which(pd$component == names(PAL)[2]), function(i) {
    r <- pd[i, ]
    xs <- seq(r$lo + 0.004, r$hi - 0.004, by = 0.007)
    if (length(xs) < 1) return(NULL)
    data.frame(x = xs, yy = r$yy)
  }))

lab_top <- pd %>% filter(yy == max(yy))

f <- ggplot() +
  geom_vline(xintercept = seq(0.02, 0.12, by = 0.02),
             colour = "grey94", linewidth = 0.3) +
  geom_rect(data = pd,
            aes(xmin = lo, xmax = hi, fill = component,
                ymin = yy - HH, ymax = yy + HH),
            colour = "white", linewidth = 0.6) +
  geom_segment(data = hatch,
               aes(x = x, xend = x,
                   y = yy - HH + 0.02,
                   yend = yy + HH - 0.02),
               colour = "white", linewidth = 0.3,
               inherit.aes = FALSE) +
  geom_text(data = pd,
            aes(x = mid, y = yy,
                label = sprintf("%.4f", cost)),
            colour = "white", fontface = "bold",
            size = 3.0, family = FONT) +
  geom_text(data = pd %>% distinct(outcome, total,
                                   ratio, yy),
            aes(x = total + 0.004, y = yy,
                label = sprintf(
                  "%.1f\u00d7 larger", ratio)),
            hjust = 0, size = 3.0, colour = "grey30",
            family = FONT) +
  geom_text(data = lab_top,
            aes(x = mid, y = yy + HH + 0.26,
                label = component, colour = component),
            fontface = "bold", size = 3.1,
            family = FONT, show.legend = FALSE) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_x_continuous(
    limits = c(0, 0.163),
    breaks = seq(0, 0.12, by = 0.02),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(OUT_ORD), labels = OUT_ORD,
    limits = c(0.5, length(OUT_ORD) + 0.95),
    expand = c(0, 0)) +
  labs(x = "Cost in AUROC",
       y = NULL,
       title = paste("The Institution Costs Three to",
                     "Four Times More Than the",
                     "Feature Restriction")) +
  theme(panel.grid = element_blank(),
        axis.text = element_text(size = 9.5),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 8)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 7...\n")
print(system.time(
  save_fig(f, "fig7_transfer_cost", 8.6, 3.6)))


















# ---------------------------------------------------------
# fig6_supervision_ladder.R
# Fusion increment as the EHR modality is strengthened
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ragg)
library(systemfonts)

fig_dir <- paste0(file.path(Sys.getenv("USERPROFILE"), "Documents", ""),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

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

PAL <- c("30-Day Death" = "#0072B2",
         "Composite (30-Day)" = "#D55E00")
SHP <- c("30-Day Death" = 16,
         "Composite (30-Day)" = 17)

RUNG <- c("Transferred\n28 features",
          "In-domain\n28 features",
          "In-domain\nall features")

d <- tribble(
  ~outcome, ~rung, ~inc, ~lo, ~hi,
  "30-Day Death", 1, 0.0262, 0.0095, 0.0419,
  "30-Day Death", 2, 0.0137, 0.0036, 0.0240,
  "30-Day Death", 3, 0.0077, 0.0001, 0.0156,
  "Composite (30-Day)", 1, 0.0212, NA, NA,
  "Composite (30-Day)", 2, 0.0138, NA, NA,
  "Composite (30-Day)", 3, 0.0065, NA, NA
)

pd <- d %>%
  mutate(outcome = factor(outcome, levels = names(PAL)),
         rung = factor(rung, levels = 1:3,
                       labels = RUNG))

cat("\n--- fusion increment by rung ---\n")
print(as.data.frame(pd), digits = 3)
cat("\nhalving at each step?\n")
print(pd %>% group_by(outcome) %>%
        summarise(r1 = inc[1] / inc[2],
                  r2 = inc[2] / inc[3],
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

lab <- pd %>% filter(rung == RUNG[1])

f <- ggplot(pd, aes(x = rung, y = inc,
                    colour = outcome, shape = outcome,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(linewidth = 0.7, alpha = 0.65) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.09,
                linewidth = 0.45, na.rm = TRUE) +
  geom_point(size = 2.9) +
  geom_text(aes(label = sprintf("%+.4f", inc)),
            vjust = -1.3, hjust = 0.5, size = 2.9,
            family = FONT, show.legend = FALSE) +
  geom_text(data = lab,
            aes(label = outcome), hjust = -0.10,
            vjust = 2.3, size = 3.1, family = FONT,
            fontface = "bold", show.legend = FALSE) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_y_continuous(
    limits = c(-0.004, 0.050),
    breaks = seq(0, 0.05, by = 0.01),
    labels = function(x) sprintf("%+.3f", x)) +
  labs(x = "Configuration of the EHR Modality",
       y = "Fusion Increment Over the EHR Modality",
       title = paste("Fusion Adds Less as the Primary",
                     "Modality Is Strengthened")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 6...\n")
print(system.time(
  save_fig(f, "fig6_supervision_ladder", 7.0, 4.2)))



















# ---------------------------------------------------------
# fig14_corr_shift.R
# Feature correlation at source vs target institution
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
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

STABLE <- "#0072B2"   # Okabe-Ito blue
SHIFT <- "#D55E00"    # Okabe-Ito vermillion

N_LAB <- 8            # how many divergences to name

NICE <- c(rr = "resp rate", hr = "heart rate",
          sbp = "systolic BP", dbp = "diastolic BP",
          bun = "urea nitrogen", aniongap = "anion gap",
          hct = "haematocrit", hgb = "haemoglobin",
          rbc = "red cells", wbc = "white cells",
          plt = "platelets", temp = "temperature",
          bicarb = "bicarbonate",
          heart_failure = "heart failure")

nice <- function(x) ifelse(x %in% names(NICE),
                           NICE[x], x)

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir, "corr_shift.csv"),
              stringsAsFactors = FALSE)

cat("\npairs:", nrow(d), "\n")
cat("mean |change|: %.4f\n")
cat(sprintf("mean |change| %.4f  max %.4f\n",
            mean(abs(d$diff)), max(abs(d$diff))))
cat(sprintf("corr of corrs %.4f\n",
            cor(d$source, d$target)))

pd <- d %>%
  mutate(pair = paste(nice(f1), "\u2013", nice(f2)),
         reverse = sign(source) != sign(target) &
           abs(source) > 0.05 & abs(target) > 0.05,
         big = rank(-abs(diff)) <= N_LAB)

cat("\nsign reversals with both |r| > 0.05:",
    sum(pd$reverse), "\n")
cat("\nlabelled pairs:\n")
print(pd %>% filter(big) %>%
        select(pair, source, target, diff) %>%
        arrange(diff) %>% as.data.frame(), digits = 3)

lim <- c(-0.62, 1.02)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = source, y = target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             shape = 16, size = 1.7, alpha = 0.55,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             shape = 22, size = 2.3, stroke = 0.8,
             colour = SHIFT, fill = "white") +
  geom_text(data = filter(pd, big),
            aes(label = pair), hjust = -0.12,
            vjust = 0.4, size = 2.7, colour = SHIFT,
            family = FONT) +
  annotate("text", x = -0.58, y = 0.96, hjust = 0,
           label = paste("hollow squares reverse sign",
                         "between institutions"),
           size = 2.9, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Correlation at the Source Institution",
       y = "Correlation at the Target Institution",
       title = paste("Physiological Relationships",
                     "Transfer; Practice-Mediated",
                     "Ones Do Not")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14...\n")
print(system.time(
  save_fig(f, "fig14_corr_shift", 6.6, 6.4)))


























  # ---------------------------------------------------------
# fig14_corr_shift.R
# Writes two figures:
#   fig14a_corr_scatter.png  overall pattern, unlabelled
#   fig14b_corr_divergence.png  the named divergences
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

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

SRC <- "#0072B2"    # source institution
TGT <- "#D55E00"    # target institution
STABLE <- "#0072B2"
REV <- "#D55E00"

N_SHOW <- 12

NICE <- c(rr = "resp rate", hr = "heart rate",
          sbp = "systolic BP", dbp = "diastolic BP",
          bun = "urea nitrogen", aniongap = "anion gap",
          hct = "haematocrit", hgb = "haemoglobin",
          rbc = "red cells", wbc = "white cells",
          plt = "platelets", temp = "temperature",
          bicarb = "bicarbonate", mchc = "MCHC",
          heart_failure = "heart failure")
nice <- function(x) ifelse(x %in% names(NICE),
                           NICE[x], x)

d <- read.csv(file.path(data_dir, "corr_shift.csv"),
              stringsAsFactors = FALSE)

cat("\npairs:", nrow(d), "\n")
cat(sprintf("mean |change| %.4f  max %.4f  corr of corrs %.4f\n",
            mean(abs(d$diff)), max(abs(d$diff)),
            cor(d$source, d$target)))

pd <- d %>%
  mutate(pair = paste(nice(f1), "\u2013", nice(f2)),
         reverse = sign(source) != sign(target) &
           abs(source) > 0.05 & abs(target) > 0.05)

cat("sign reversals with both |r| > 0.05:",
    sum(pd$reverse), "\n")

## ---- FIGURE 14a: the overall pattern ---------------------

lim <- c(-0.62, 1.02)

f1 <- ggplot(pd, aes(x = source, y = target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey90",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey90",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             shape = 16, size = 1.7, alpha = 0.5,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             shape = 22, size = 2.4, stroke = 0.85,
             colour = REV, fill = "white") +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign",
                         "between institutions"),
           size = 2.9, colour = "grey35",
           family = FONT) +
  annotate("text", x = 0.98, y = 0.86, hjust = 1,
           label = "strong relationships preserved",
           size = 2.9, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Correlation at the Source Institution",
       y = "Correlation at the Target Institution",
       title = paste("Most Feature Relationships Are",
                     "Preserved Across Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14a...\n")
print(system.time(
  save_fig(f1, "fig14a_corr_scatter", 6.2, 6.0)))

## ---- FIGURE 14b: the named divergences -------------------

top <- pd %>%
  slice_max(abs(diff), n = N_SHOW) %>%
  arrange(diff) %>%
  mutate(pair = factor(pair, levels = pair))

cat("\n--- largest divergences ---\n")
print(as.data.frame(
  top %>% select(pair, source, target, diff)),
  digits = 3)

f2 <- ggplot(top, aes(y = pair)) +
  geom_vline(xintercept = 0, colour = "grey55",
             linewidth = 0.45) +
  geom_segment(aes(x = source, xend = target,
                   yend = pair),
               colour = "grey60", linewidth = 0.7) +
  geom_point(aes(x = source), shape = 16, size = 2.7,
             colour = SRC) +
  geom_point(aes(x = target), shape = 17, size = 2.7,
             colour = TGT) +
  geom_text(aes(x = pmin(source, target) - 0.02,
                label = sprintf("%+.3f",
                                pmin(source, target))),
            hjust = 1, size = 2.7, colour = "grey35",
            family = FONT) +
  geom_text(aes(x = pmax(source, target) + 0.02,
                label = sprintf("%+.3f",
                                pmax(source, target))),
            hjust = 0, size = 2.7, colour = "grey35",
            family = FONT) +
  annotate("text", x = -0.30, y = N_SHOW + 0.55,
           hjust = 0, label = "source", size = 3.1,
           colour = SRC, fontface = "bold",
           family = FONT) +
  annotate("text", x = -0.12, y = N_SHOW + 0.55,
           hjust = 0, label = "target", size = 3.1,
           colour = TGT, fontface = "bold",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.50, 0.52),
    breaks = seq(-0.4, 0.4, by = 0.2),
    labels = function(x) sprintf("%+.1f", x)) +
  scale_y_discrete(
    expand = expansion(add = c(0.6, 1.2))) +
  labs(x = "Pearson Correlation", y = NULL,
       title = paste("The Twelve Relationships That",
                     "Change Most Between Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey95", linewidth = 0.3,
          linetype = "dotted"),
        panel.grid.major.x = element_line(
          colour = "grey93", linewidth = 0.3),
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 7)),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 10, 4, 6))

cat("\nwriting figure 14b...\n")
print(system.time(
  save_fig(f2, "fig14b_corr_divergence", 7.4, 5.4)))






# ---------------------------------------------------------
# fig14_corr_shift.R
# Writes two figures:
#   fig14a_corr_scatter.png  overall pattern, unlabelled
#   fig14b_corr_divergence.png  the named divergences
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

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

SRC <- "#0072B2"    # source institution
TGT <- "#D55E00"    # target institution
STABLE <- "#0072B2"
REV <- "#D55E00"

N_SHOW <- 12

NICE <- c(rr = "resp rate", hr = "heart rate",
          sbp = "systolic BP", dbp = "diastolic BP",
          bun = "urea nitrogen", aniongap = "anion gap",
          hct = "haematocrit", hgb = "haemoglobin",
          rbc = "red cells", wbc = "white cells",
          plt = "platelets", temp = "temperature",
          bicarb = "bicarbonate", mchc = "MCHC",
          heart_failure = "heart failure")
nice <- function(x) ifelse(x %in% names(NICE),
                           NICE[x], x)

d <- read.csv(file.path(data_dir, "corr_shift.csv"),
              stringsAsFactors = FALSE)

cat("\npairs:", nrow(d), "\n")
cat(sprintf("mean |change| %.4f  max %.4f  corr of corrs %.4f\n",
            mean(abs(d$diff)), max(abs(d$diff)),
            cor(d$source, d$target)))

pd <- d %>%
  mutate(pair = paste(nice(f1), "\u2013", nice(f2)),
         reverse = sign(source) != sign(target) &
           abs(source) > 0.05 & abs(target) > 0.05)

cat("sign reversals with both |r| > 0.05:",
    sum(pd$reverse), "\n")

## ---- FIGURE 14a: the overall pattern ---------------------

lim <- c(-0.62, 1.02)

f1 <- ggplot(pd, aes(x = source, y = target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey90",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey90",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             shape = 16, size = 1.7, alpha = 0.5,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             shape = 22, size = 2.4, stroke = 0.85,
             colour = REV, fill = "white") +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign",
                         "between institutions"),
           size = 2.9, colour = "grey35",
           family = FONT) +
  annotate("text", x = 0.98, y = 0.86, hjust = 1,
           label = "strong relationships preserved",
           size = 2.9, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Correlation at the Source Institution",
       y = "Correlation at the Target Institution",
       title = paste("Most Feature Relationships Are",
                     "Preserved Across Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14a...\n")
print(system.time(
  save_fig(f1, "fig14a_corr_scatter", 6.2, 6.0)))

## ---- FIGURE 14b: the named divergences -------------------

top <- pd %>%
  slice_max(abs(diff), n = N_SHOW) %>%
  arrange(diff) %>%
  mutate(pair = factor(pair, levels = pair))

cat("\n--- largest divergences ---\n")
print(as.data.frame(
  top %>% select(pair, source, target, diff)),
  digits = 3)

f2 <- ggplot(top, aes(y = pair)) +
  geom_vline(xintercept = 0, colour = "grey55",
             linewidth = 0.45) +
  geom_segment(aes(x = source, xend = target,
                   yend = pair),
               colour = "grey60", linewidth = 0.7) +
  geom_point(aes(x = source), shape = 16, size = 2.7,
             colour = SRC) +
  geom_point(aes(x = target), shape = 17, size = 2.7,
             colour = TGT) +
  geom_text(aes(x = pmin(source, target) - 0.02,
                label = sprintf("%+.3f",
                                pmin(source, target))),
            hjust = 1, size = 2.7, colour = "grey35",
            family = FONT) +
  geom_text(aes(x = pmax(source, target) + 0.02,
                label = sprintf("%+.3f",
                                pmax(source, target))),
            hjust = 0, size = 2.7, colour = "grey35",
            family = FONT) +
  annotate("text", x = -0.30, y = N_SHOW + 0.55,
           hjust = 0, label = "source", size = 3.1,
           colour = SRC, fontface = "bold",
           family = FONT) +
  annotate("text", x = -0.12, y = N_SHOW + 0.55,
           hjust = 0, label = "target", size = 3.1,
           colour = TGT, fontface = "bold",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.50, 0.52),
    breaks = seq(-0.4, 0.4, by = 0.2),
    labels = function(x) sprintf("%+.1f", x)) +
  scale_y_discrete(
    expand = expansion(add = c(0.6, 1.2))) +
  labs(x = "Pearson Correlation", y = NULL,
       title = paste("The Twelve Relationships That",
                     "Change Most Between Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey95", linewidth = 0.3,
          linetype = "dotted"),
        panel.grid.major.x = element_line(
          colour = "grey93", linewidth = 0.3),
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 7)),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 10, 4, 6))

cat("\nwriting figure 14b...\n")
print(system.time(
  save_fig(f2, "fig14b_corr_divergence", 7.4, 5.4)))





























# ---------------------------------------------------------
# fig14_corr_shift.R
# Writes two figures:
#   fig14a_corr_scatter.png  overall pattern, unlabelled
#   fig14b_corr_divergence.png  the named divergences
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

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

SRC <- "#0072B2"    # source institution
TGT <- "#D55E00"    # target institution
STABLE <- "#0072B2"
REV <- "#D55E00"

N_SHOW <- 12

NICE <- c(rr = "resp rate", hr = "heart rate",
          sbp = "systolic BP", dbp = "diastolic BP",
          bun = "urea nitrogen", aniongap = "anion gap",
          hct = "haematocrit", hgb = "haemoglobin",
          rbc = "red cells", wbc = "white cells",
          plt = "platelets", temp = "temperature",
          bicarb = "bicarbonate", mchc = "MCHC",
          heart_failure = "heart failure")
nice <- function(x) ifelse(x %in% names(NICE),
                           NICE[x], x)

d <- read.csv(file.path(data_dir, "corr_shift.csv"),
              stringsAsFactors = FALSE)

cat("\npairs:", nrow(d), "\n")
cat(sprintf("mean |change| %.4f  max %.4f  corr of corrs %.4f\n",
            mean(abs(d$diff)), max(abs(d$diff)),
            cor(d$source, d$target)))

pd <- d %>%
  mutate(pair = paste(nice(f1), "\u2013", nice(f2)),
         reverse = sign(source) != sign(target) &
           abs(source) > 0.05 & abs(target) > 0.05)

cat("sign reversals with both |r| > 0.05:",
    sum(pd$reverse), "\n")

## ---- FIGURE 14a: the overall pattern ---------------------

lim <- c(-0.62, 1.02)

f1 <- ggplot(pd, aes(x = source, y = target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey90",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey90",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             shape = 16, size = 1.7, alpha = 0.5,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             shape = 22, size = 2.4, stroke = 0.85,
             colour = REV, fill = "white") +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign",
                         "between institutions"),
           size = 2.9, colour = "grey35",
           family = FONT) +
  annotate("text", x = 0.98, y = 0.86, hjust = 1,
           label = "strong relationships preserved",
           size = 2.9, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.5),
    labels = function(x) sprintf("%+.1f", x),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Correlation at the Source Institution",
       y = "Correlation at the Target Institution",
       title = paste("Most Feature Relationships Are",
                     "Preserved Across Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14a...\n")
print(system.time(
  save_fig(f1, "fig14a_corr_scatter", 6.2, 6.0)))

## ---- FIGURE 14b: the named divergences -------------------

top <- pd %>%
  slice_max(abs(diff), n = N_SHOW) %>%
  arrange(diff) %>%
  mutate(pair = factor(pair, levels = pair))

cat("\n--- largest divergences ---\n")
print(as.data.frame(
  top %>% select(pair, source, target, diff)),
  digits = 3)

f2 <- ggplot(top, aes(y = pair)) +
  geom_vline(xintercept = 0, colour = "grey55",
             linewidth = 0.45) +
  geom_segment(aes(x = source, xend = target,
                   yend = pair),
               colour = "grey60", linewidth = 0.7) +
  geom_point(aes(x = source), shape = 16, size = 2.7,
             colour = SRC) +
  geom_point(aes(x = target), shape = 17, size = 2.7,
             colour = TGT) +
  geom_text(aes(x = pmin(source, target) - 0.02,
                label = sprintf("%+.3f",
                                pmin(source, target))),
            hjust = 1, size = 2.7, colour = "grey35",
            family = FONT) +
  geom_text(aes(x = pmax(source, target) + 0.02,
                label = sprintf("%+.3f",
                                pmax(source, target))),
            hjust = 0, size = 2.7, colour = "grey35",
            family = FONT) +
  annotate("text", x = -0.30, y = N_SHOW + 0.55,
           hjust = 0, label = "source", size = 3.1,
           colour = SRC, fontface = "bold",
           family = FONT) +
  annotate("text", x = -0.12, y = N_SHOW + 0.55,
           hjust = 0, label = "target", size = 3.1,
           colour = TGT, fontface = "bold",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.50, 0.52),
    breaks = seq(-0.4, 0.4, by = 0.2),
    labels = function(x) sprintf("%+.1f", x)) +
  scale_y_discrete(
    expand = expansion(add = c(0.6, 1.2))) +
  labs(x = "Pearson Correlation", y = NULL,
       title = paste("The Twelve Relationships That",
                     "Change Most Between Institutions")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey95", linewidth = 0.3,
          linetype = "dotted"),
        panel.grid.major.x = element_line(
          colour = "grey93", linewidth = 0.3),
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 7)),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 10, 4, 6))

cat("\nwriting figure 14b...\n")
print(system.time(
  save_fig(f2, "fig14b_corr_divergence", 7.4, 5.4)))



install.packages("ggrepel")
library(ggrepel)























# ---------------------------------------------------------
# fig14_corr_shift.R
# Feature correlation at source vs target institution
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ggrepel)
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

STABLE <- "#0072B2"   # Okabe-Ito blue
SHIFT <- "#D55E00"    # Okabe-Ito vermillion

N_LAB <- 8            # divergences to name
REV_MIN <- 0.03       # |r| floor for a sign reversal

NICE <- c(rr = "resp rate", hr = "heart rate",
          sbp = "systolic BP", dbp = "diastolic BP",
          bun = "urea nitrogen", aniongap = "anion gap",
          hct = "haematocrit", hgb = "haemoglobin",
          rbc = "red cells", wbc = "white cells",
          plt = "platelets", temp = "temperature",
          bicarb = "bicarbonate",
          heart_failure = "heart failure",
          creatinine = "creatinine",
          potassium = "potassium",
          chloride = "chloride", calcium = "calcium")

nice <- function(x) ifelse(x %in% names(NICE),
                           NICE[x], x)

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir, "corr_shift.csv"),
              stringsAsFactors = FALSE)

cat("\npairs:", nrow(d), "\n")
cat(sprintf("mean |change| %.4f   max %.4f\n",
            mean(abs(d$diff)), max(abs(d$diff))))
cat(sprintf("corr of corrs %.4f\n",
            cor(d$source, d$target)))

pd <- d %>%
  mutate(pair = paste(nice(f1), "\u2013", nice(f2)),
         reverse = sign(source) != sign(target) &
           abs(source) > REV_MIN &
           abs(target) > REV_MIN,
         big = rank(-abs(diff),
                    ties.method = "first") <= N_LAB)

cat("\nsign reversals with both |r| >",
    REV_MIN, ":", sum(pd$reverse), "\n")
cat("labelled and reversing:",
    sum(pd$big & pd$reverse), "of", N_LAB, "\n\n")
print(pd %>% filter(big) %>%
        select(pair, source, target, diff, reverse) %>%
        arrange(diff) %>% as.data.frame(), digits = 3)

lim <- c(-0.62, 1.02)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = source, y = target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             shape = 16, size = 1.7, alpha = 0.5,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             shape = 22, size = 2.4, stroke = 0.85,
             colour = SHIFT, fill = "white") +
  geom_text_repel(
    data = filter(pd, big),
    aes(label = pair),
    size = 2.8, colour = SHIFT, family = FONT,
    segment.colour = "grey60",
    segment.size = 0.3,
    min.segment.length = 0,
    box.padding = 0.45,
    point.padding = 0.30,
    force = 6, max.overlaps = Inf,
    seed = 42) +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign",
                         "between institutions"),
           size = 2.9, colour = "grey35",
           family = FONT) +
  annotate("text", x = 0.62, y = -0.50, hjust = 0,
           label = "points on the line are unchanged",
           size = 2.9, colour = "grey45",
           family = FONT) +
  scale_x_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Correlation at the Source Institution",
       y = "Correlation at the Target Institution",
       title = paste("Physiological Relationships",
                     "Transfer; Practice-Mediated",
                     "Ones Do Not")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14...\n")
print(system.time(
  save_fig(f, "fig14_corr_shift", 7.0, 6.8)))

























  # ---------------------------------------------------------
# fig14_corr_shift.R
# Feature correlation at source vs target institution
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ggrepel)
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

STABLE <- "#0072B2"   # Okabe-Ito blue
SHIFT <- "#D55E00"    # Okabe-Ito vermillion

N_LAB <- 8            # divergences to name
REV_MIN <- 0.03       # magnitude floor for a reversal

NICE <- c(rr = "resp rate", hr = "heart rate",
          sbp = "systolic BP", dbp = "diastolic BP",
          bun = "urea nitrogen", aniongap = "anion gap",
          hct = "haematocrit", hgb = "haemoglobin",
          rbc = "red cells", wbc = "white cells",
          plt = "platelets", temp = "temperature",
          bicarb = "bicarbonate",
          heart_failure = "heart failure",
          creatinine = "creatinine",
          potassium = "potassium",
          chloride = "chloride", calcium = "calcium")

nice <- function(x) {
  out <- unname(NICE[x])
  ifelse(is.na(out), x, out)
}

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir, "corr_shift.csv"),
              stringsAsFactors = FALSE)

cat("\npairs:", nrow(d), "\n")
cat(sprintf("mean abs change %.4f   max %.4f\n",
            mean(abs(d$diff)), max(abs(d$diff))))
cat(sprintf("corr of corrs %.4f\n",
            cor(d$source, d$target)))

pd <- d %>%
  mutate(pair = paste(nice(f1), "-", nice(f2)),
         reverse = sign(source) != sign(target) &
           abs(source) > REV_MIN &
           abs(target) > REV_MIN,
         big = rank(-abs(diff),
                    ties.method = "first") <= N_LAB)

cat("\nsign reversals above the floor:",
    sum(pd$reverse), "\n")
cat("labelled and reversing:",
    sum(pd$big & pd$reverse), "of", N_LAB, "\n\n")
print(pd %>% filter(big) %>%
        select(pair, source, target, diff, reverse) %>%
        arrange(diff) %>% as.data.frame(), digits = 3)

lim <- c(-0.62, 1.02)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = source, y = target)) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             shape = 16, size = 1.7, alpha = 0.5,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             shape = 22, size = 2.4, stroke = 0.85,
             colour = SHIFT, fill = "white") +
  geom_text_repel(
    data = filter(pd, big),
    aes(label = pair),
    size = 2.8, colour = SHIFT, family = FONT,
    segment.colour = "grey60",
    segment.size = 0.3,
    min.segment.length = 0,
    box.padding = 0.45,
    point.padding = 0.30,
    force = 6, max.overlaps = Inf,
    seed = 42) +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign",
                         "between institutions"),
           size = 2.9, colour = "grey35",
           family = FONT) +
  annotate("text", x = 0.60, y = -0.50, hjust = 0,
           label = "points on the line are unchanged",
           size = 2.9, colour = "grey45",
           family = FONT) +
  scale_x_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = lim, breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Correlation at the Source Institution",
       y = "Correlation at the Target Institution",
       title = paste("Physiological Relationships",
                     "Transfer; Practice-Mediated",
                     "Ones Do Not")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14...\n")
print(system.time(
  save_fig(f, "fig14_corr_shift", 7.0, 6.8)))
