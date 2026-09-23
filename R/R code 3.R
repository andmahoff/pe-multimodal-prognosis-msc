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

N_LAB <- 8
REV_MIN <- 0.03

## data region ends at 1.02; margin runs beyond it
X_MAX <- 1.02
X_TICK <- 1.07        # leader lines converge here
X_TEXT <- 1.10        # label column starts here
X_LIM <- 1.72         # trimmed to fit the longest label

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

lab <- pd %>%
  filter(big) %>%
  arrange(desc(target)) %>%
  mutate(ty = seq(0.72, -0.44, length.out = n()),
         txt = sprintf("%s\n%+.2f to %+.2f",
                       pair, source, target))

print(lab %>% select(pair, source, target, diff,
                     reverse) %>%
        as.data.frame(), digits = 3)

Y_LIM <- c(-0.62, 1.02)

## ---- 3. plot and save -----------------------------------

f <- ggplot() +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             aes(x = source, y = target),
             shape = 16, size = 1.7, alpha = 0.45,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             aes(x = source, y = target),
             shape = 22, size = 2.5, stroke = 0.9,
             colour = SHIFT, fill = "white") +
  geom_vline(xintercept = X_MAX, colour = "grey80",
             linewidth = 0.4) +
  geom_segment(data = lab,
               aes(x = source, y = target,
                   xend = X_TICK, yend = ty),
               colour = "grey65", linewidth = 0.28) +
  geom_point(data = lab,
             aes(x = X_TICK, y = ty),
             shape = 22, size = 1.8, stroke = 0.7,
             colour = SHIFT, fill = "white") +
  geom_text(data = lab,
            aes(x = X_TEXT, y = ty, label = txt),
            hjust = 0, vjust = 0.5, size = 2.75,
            lineheight = 0.95, colour = SHIFT,
            family = FONT) +
  annotate("text", x = X_TEXT, y = 0.94, hjust = 0,
           label = "Eight largest divergences",
           size = 2.95, fontface = "bold",
           colour = "grey20", family = FONT) +
  annotate("text", x = X_TEXT, y = 0.86, hjust = 0,
           label = "source to target",
           size = 2.65, colour = "grey45",
           family = FONT) +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign;",
                         "points on the line are",
                         "unchanged"),
           size = 2.8, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.62, X_LIM),
    breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = Y_LIM,
    breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
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
          size = 12, face = "bold", hjust = 0.34,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

cat("\nwriting figure 14...\n")
print(system.time(
  save_fig(f, "fig14_corr_shift", 8.4, 6.2)))














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

N_LAB <- 8            # divergences to name
REV_MIN <- 0.03       # magnitude floor for a reversal

## data region ends at 1.02; margin runs beyond it
X_MAX <- 1.02
X_TICK <- 1.10        # leader lines converge here
X_TEXT <- 1.14        # label column starts here
X_LIM <- 2.05         # full canvas including margin

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

lab <- pd %>%
  filter(big) %>%
  arrange(desc(target)) %>%
  mutate(ty = seq(0.60, -0.42,
                  length.out = n()),
         txt = sprintf("%s  (%+.2f to %+.2f)",
                       pair, source, target))

print(lab %>% select(pair, source, target, diff,
                     reverse) %>%
        as.data.frame(), digits = 3)

Y_LIM <- c(-0.62, 1.02)

## ---- 3. plot and save -----------------------------------

f <- ggplot() +
  ## data region
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             aes(x = source, y = target),
             shape = 16, size = 1.7, alpha = 0.45,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             aes(x = source, y = target),
             shape = 22, size = 2.5, stroke = 0.9,
             colour = SHIFT, fill = "white") +
  ## boundary between data region and label margin
  geom_vline(xintercept = X_MAX, colour = "grey80",
             linewidth = 0.4) +
  ## leader lines: point -> converge -> label
  geom_segment(data = lab,
               aes(x = source, y = target,
                   xend = X_TICK, yend = ty),
               colour = "grey65", linewidth = 0.28) +
  geom_point(data = lab,
             aes(x = X_TICK, y = ty),
             shape = 22, size = 1.8, stroke = 0.7,
             colour = SHIFT, fill = "white") +
  geom_text(data = lab,
            aes(x = X_TEXT, y = ty, label = txt),
            hjust = 0, size = 2.85, colour = SHIFT,
            family = FONT) +
  annotate("text", x = X_TEXT, y = 0.76, hjust = 0,
           label = "Eight largest divergences",
           size = 3.0, fontface = "bold",
           colour = "grey20", family = FONT) +
  annotate("text", x = X_TEXT, y = 0.68, hjust = 0,
           label = "source to target correlation",
           size = 2.7, colour = "grey45",
           family = FONT) +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign;",
                         "points on the line are",
                         "unchanged"),
           size = 2.85, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.62, X_LIM),
    breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = Y_LIM,
    breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
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
          size = 12, face = "bold", hjust = 0.28,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 14...\n")
print(system.time(
  save_fig(f, "fig14_corr_shift", 10.4, 6.4)))




















# ---------------------------------------------------------
# fig7_transfer_cost.R
# Feature-restriction cost vs institution cost
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

PAL <- c("Restricting the feature set" = "#9ECAE1",
         "Training at another institution" = "#08519C")
TXT <- c("Restricting the feature set" = "#2171B5",
         "Training at another institution" = "#08306B")

N_RULE <- 4     # fixed rules per institution segment

## ---- 2. data --------------------------------------------

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
    xs <- seq(r$lo, r$hi,
              length.out = N_RULE + 2)
    xs <- xs[c(-1, -length(xs))]
    data.frame(x = xs, yy = r$yy)
  }))

lab_top <- pd %>% filter(yy == max(yy))

## ---- 3. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.02, 0.12, by = 0.02),
             colour = "grey93", linewidth = 0.3) +
  geom_rect(data = pd,
            aes(xmin = lo, xmax = hi, fill = component,
                ymin = yy - HH, ymax = yy + HH),
            colour = "grey25", linewidth = 0.3) +
  geom_segment(data = hatch,
               aes(x = x, xend = x,
                   y = yy - HH + 0.015,
                   yend = yy + HH - 0.015),
               colour = "white", linewidth = 0.35,
               inherit.aes = FALSE) +
  geom_text(data = filter(pd,
                          component == names(PAL)[1]),
            aes(x = mid, y = yy,
                label = sprintf("%.4f", cost)),
            colour = "grey15", fontface = "bold",
            size = 3.0, family = FONT) +
  geom_text(data = filter(pd,
                          component == names(PAL)[2]),
            aes(x = mid, y = yy,
                label = sprintf("%.4f", cost)),
            colour = "white", fontface = "bold",
            size = 3.0, family = FONT) +
  geom_text(data = pd %>% distinct(outcome, total,
                                   ratio, yy),
            aes(x = total + 0.003, y = yy,
                label = sprintf("%.1f\u00d7", ratio)),
            hjust = 0, fontface = "bold", size = 3.1,
            colour = "grey25", family = FONT) +
  geom_text(data = lab_top,
            aes(x = mid, y = yy + HH + 0.23,
                label = component, colour = component),
            fontface = "bold", size = 3.1,
            family = FONT, show.legend = FALSE) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = TXT, guide = "none") +
  scale_x_continuous(
    limits = c(0, 0.142),
    breaks = seq(0, 0.12, by = 0.02),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(OUT_ORD), labels = OUT_ORD,
    limits = c(0.5, length(OUT_ORD) + 0.85),
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
  save_fig(f, "fig7_transfer_cost", 7.8, 3.4)))




























# ---------------------------------------------------------
# fig6_supervision_ladder.R
# Fusion increment as the EHR modality is strengthened
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

PAL <- c("30-Day Death" = "#0072B2",
         "Composite (30-Day)" = "#D55E00")
SHP <- c("30-Day Death" = 16,
         "Composite (30-Day)" = 17)
LTY <- c("30-Day Death" = "solid",
         "Composite (30-Day)" = "22")

RUNG <- c("Transferred\n28 features",
          "In-domain\n28 features",
          "In-domain\nall features")

DODGE <- 0.10   # horizontal offset between series

## ---- 2. data --------------------------------------------

d <- tribble(
  ~outcome, ~rung, ~inc, ~lo, ~hi, ~vj,
  "30-Day Death", 1, 0.0262, 0.0095, 0.0419, -1.4,
  "30-Day Death", 2, 0.0137, 0.0036, 0.0240, -1.4,
  "30-Day Death", 3, 0.0077, 0.0001, 0.0156, -1.4,
  "Composite (30-Day)", 1, 0.0212, NA, NA,  2.1,
  "Composite (30-Day)", 2, 0.0138, NA, NA,  2.1,
  "Composite (30-Day)", 3, 0.0065, NA, NA,  2.1
)

pd <- d %>%
  mutate(outcome = factor(outcome,
                          levels = names(PAL)),
         x = rung + ifelse(
           outcome == "30-Day Death",
           -DODGE, DODGE))

cat("\n--- fusion increment by rung ---\n")
print(as.data.frame(pd %>% select(-vj, -x)),
      digits = 3)
cat("\nratio between successive rungs:\n")
print(pd %>% group_by(outcome) %>%
        summarise(r1 = inc[1] / inc[2],
                  r2 = inc[2] / inc[3],
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

## series labels placed clear of the data
lab <- tibble(
  outcome = factor(names(PAL), levels = names(PAL)),
  x = c(2.02, 2.02),
  y = c(0.0225, 0.0043))

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = x, y = inc,
                    colour = outcome,
                    shape = outcome,
                    linetype = outcome,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(linewidth = 0.7) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.06,
                linewidth = 0.45, linetype = "solid",
                na.rm = TRUE) +
  geom_point(size = 2.9) +
  geom_text(aes(label = sprintf("%+.4f", inc),
                vjust = vj),
            hjust = 0.5, size = 2.9, family = FONT,
            show.legend = FALSE) +
  geom_text(data = lab,
            aes(x = x, y = y, label = outcome,
                colour = outcome),
            hjust = 0, size = 3.1, fontface = "bold",
            family = FONT, inherit.aes = FALSE,
            show.legend = FALSE) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY,
                        guide = "none") +
  scale_x_continuous(
    breaks = 1:3, labels = RUNG,
    limits = c(0.65, 3.35), expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(-0.002, 0.046),
    breaks = seq(0, 0.04, by = 0.01),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Configuration of the EHR Modality",
       y = "Fusion Increment Over the EHR Modality",
       title = paste("Fusion Adds Less as the Primary",
                     "Modality Is Strengthened")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text.x = element_text(size = 9,
                                   lineheight = 0.95),
        axis.text.y = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 6...\n")
print(system.time(
  save_fig(f, "fig6_supervision_ladder", 7.2, 4.2)))


























# ---------------------------------------------------------
# fig6_supervision_ladder.R
# Fusion increment as the EHR modality is strengthened
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

PAL <- c("30-Day Death" = "#0072B2",
         "Composite (30-Day)" = "#D55E00")
SHP <- c("30-Day Death" = 16,
         "Composite (30-Day)" = 17)
LTY <- c("30-Day Death" = "solid",
         "Composite (30-Day)" = "22")

RUNG <- c("Transferred\n28 features",
          "In-domain\n28 features",
          "In-domain\nall features")

## ---- 2. data --------------------------------------------

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
  mutate(outcome = factor(outcome,
                          levels = names(PAL)),
         rung = factor(rung, levels = 1:3,
                       labels = RUNG))

cat("\n--- fusion increment by rung ---\n")
print(as.data.frame(pd), digits = 3)
cat("\nratio between successive rungs:\n")
print(pd %>% group_by(outcome) %>%
        summarise(r1 = inc[1] / inc[2],
                  r2 = inc[2] / inc[3],
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = rung, y = inc,
                    colour = outcome,
                    shape = outcome,
                    linetype = outcome,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(linewidth = 0.75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.10,
                linewidth = 0.45, linetype = "solid",
                na.rm = TRUE) +
  geom_point(size = 3.0) +
  geom_text(aes(label = sprintf("%+.4f", inc)),
            vjust = -1.5, hjust = 0.5, size = 3.0,
            fontface = "bold", family = FONT,
            show.legend = FALSE) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY,
                        guide = "none") +
  scale_x_discrete(expand = expansion(add = 0.42)) +
  scale_y_continuous(
    limits = c(-0.002, 0.047),
    breaks = seq(0, 0.04, by = 0.01),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Configuration of the EHR Modality",
       y = "Fusion Increment Over the EHR Modality",
       title = paste("Fusion Adds Less as the Primary",
                     "Modality Is Strengthened")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.x = unit(1.0, "lines"),
        strip.text = element_text(size = 10,
                                  face = "bold"),
        axis.text.x = element_text(size = 8.5,
                                   lineheight = 0.95),
        axis.text.y = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 12)),
        axis.title.y = element_text(
          size = 9, margin = margin(r = 6)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 6...\n")
print(system.time(
  save_fig(f, "fig6_supervision_ladder", 8.0, 4.2)))

























# ---------------------------------------------------------
# fig6_supervision_ladder.R
# Fusion increment as the EHR modality is strengthened
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

PAL <- c("30-Day Death" = "#0072B2",
         "Composite (30-Day)" = "#D55E00")
SHP <- c("30-Day Death" = 16,
         "Composite (30-Day)" = 17)
LTY <- c("30-Day Death" = "solid",
         "Composite (30-Day)" = "22")

RUNG <- c("Transferred\n28 features",
          "In-domain\n28 features",
          "In-domain\nall features")

NUDGE <- 0.20   # horizontal offset for value labels

## ---- 2. data --------------------------------------------

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
  mutate(outcome = factor(outcome,
                          levels = names(PAL)),
         rungn = rung,
         rung = factor(rung, levels = 1:3,
                       labels = RUNG),
         lx = ifelse(rungn == 3,
                     rungn - NUDGE, rungn + NUDGE),
         hj = ifelse(rungn == 3, 1, 0))

cat("\n--- fusion increment by rung ---\n")
print(as.data.frame(pd %>%
  select(outcome, rung, inc, lo, hi)), digits = 3)
cat("\nratio between successive rungs:\n")
print(pd %>% group_by(outcome) %>%
        summarise(r1 = inc[1] / inc[2],
                  r2 = inc[2] / inc[3],
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = rungn, y = inc,
                    colour = outcome,
                    shape = outcome,
                    linetype = outcome,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(linewidth = 0.75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.10,
                linewidth = 0.45, linetype = "solid",
                na.rm = TRUE) +
  geom_point(size = 3.0) +
  geom_text(aes(x = lx, y = inc, hjust = hj,
                label = sprintf("%+.4f", inc)),
            vjust = 0.5, size = 3.0,
            fontface = "bold", family = FONT,
            show.legend = FALSE) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY,
                        guide = "none") +
  scale_x_continuous(
    breaks = 1:3, labels = RUNG,
    limits = c(0.62, 3.38), expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(-0.002, 0.046),
    breaks = seq(0, 0.04, by = 0.01),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Configuration of the EHR Modality",
       y = "Fusion Increment Over the EHR Modality",
       title = paste("Fusion Adds Less as the Primary",
                     "Modality Is Strengthened")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.x = unit(1.0, "lines"),
        strip.text = element_text(size = 10,
                                  face = "bold"),
        axis.text.x = element_text(size = 8.5,
                                   lineheight = 0.95),
        axis.text.y = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 12)),
        axis.title.y = element_text(
          size = 9, margin = margin(r = 6)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 6...\n")
print(system.time(
  save_fig(f, "fig6_supervision_ladder", 8.2, 4.2)))




























# ---------------------------------------------------------
# fig3_fusion_schematic.R
# Where each fusion approach intervenes in the pipeline
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

theme_set(theme_void(base_size = 10,
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

## Okabe-Ito
BLUE <- "#0072B2"
ORANGE <- "#D55E00"
GREEN <- "#009E73"
PURPLE <- "#CC79A7"
GREY <- "#4D4D4D"
FAINT <- "#F2F2F2"

## ---- 2. pipeline stages ---------------------------------

STAGE <- tribble(
  ~x, ~lab,
  1,  "Raw features",
  2,  "Per-modality\nmodel",
  3,  "Modality\nprediction",
  4,  "Combination",
  5,  "Fused score"
)

MOD <- c("EHR", "ECG", "CTPA")
YM <- c(3.05, 2.15, 1.25)      # one row per modality

BW <- 0.34                      # box half-width
BH <- 0.26                      # box half-height

boxes <- expand.grid(x = 1:3,
                     i = seq_along(MOD)) %>%
  mutate(y = YM[i], lab = MOD[i]) %>%
  filter(!(x == 3))             # col 3 drawn separately

pred <- tibble(x = 3, y = YM, lab = "p")

## ---- 3. intervention points -----------------------------
## each method labelled where it enters the pipeline

METH <- tribble(
  ~x,   ~y,   ~lab,                      ~col,   ~lw,
  1.5,  3.85, "Early fusion",            ORANGE, 0.8,
  2.5,  3.85, "Joint representation",    PURPLE, 0.8,
  4.0,  3.85, "Weighted / equal average", BLUE,  1.0,
  4.0,  0.45, "Meta-learners",           GREEN,  0.8,
  4.0,  0.05, "Conditional gating",      GREY,   0.8
)

cat("\npipeline stages:", nrow(STAGE), "\n")
cat("methods labelled:", nrow(METH), "\n")
print(as.data.frame(METH[, c("lab", "x")]))

## ---- 4. plot --------------------------------------------

f <- ggplot() +

  ## stage band
  annotate("rect", xmin = 0.55, xmax = 5.45,
           ymin = 0.85, ymax = 3.45,
           fill = FAINT, colour = NA) +

  ## flow arrows between stages, per modality
  annotate("segment",
           x = rep(c(1.36, 2.36), each = 3),
           xend = rep(c(1.64, 2.64), each = 3),
           y = rep(YM, 2), yend = rep(YM, 2),
           colour = "grey55", linewidth = 0.4,
           arrow = arrow(length = unit(0.07, "in"),
                         type = "closed")) +

  ## converge into the combination node
  annotate("segment",
           x = rep(3.36, 3), xend = rep(3.86, 3),
           y = YM, yend = rep(2.15, 3),
           colour = "grey55", linewidth = 0.4,
           arrow = arrow(length = unit(0.07, "in"),
                         type = "closed")) +

  ## combination -> fused score
  annotate("segment", x = 4.34, xend = 4.66,
           y = 2.15, yend = 2.15,
           colour = "grey55", linewidth = 0.4,
           arrow = arrow(length = unit(0.07, "in"),
                         type = "closed")) +

  ## modality boxes, columns 1 and 2
  geom_rect(data = boxes,
            aes(xmin = x - BW, xmax = x + BW,
                ymin = y - BH, ymax = y + BH),
            fill = "white", colour = "grey35",
            linewidth = 0.4) +
  geom_text(data = boxes,
            aes(x = x, y = y, label = lab),
            size = 3.1, family = FONT,
            colour = "grey15") +

  ## prediction nodes, column 3
  geom_point(data = pred,
             aes(x = x, y = y),
             shape = 21, size = 5.4, stroke = 0.5,
             fill = "white", colour = "grey35") +

  ## combination node
  annotate("rect", xmin = 4 - 0.34, xmax = 4 + 0.34,
           ymin = 2.15 - 0.30, ymax = 2.15 + 0.30,
           fill = "white", colour = BLUE,
           linewidth = 1.0) +
  annotate("text", x = 4, y = 2.15,
           label = "combine", size = 3.1,
           family = FONT, colour = BLUE,
           fontface = "bold") +

  ## fused score node
  annotate("rect", xmin = 5 - 0.34, xmax = 5 + 0.34,
           ymin = 2.15 - 0.26, ymax = 2.15 + 0.26,
           fill = "white", colour = "grey35",
           linewidth = 0.4) +
  annotate("text", x = 5, y = 2.15, label = "score",
           size = 3.1, family = FONT,
           colour = "grey15") +

  ## intervention markers: dashed drop lines
  annotate("segment",
           x = METH$x, xend = METH$x,
           y = ifelse(METH$y > 3.4, 3.45, 0.85),
           yend = ifelse(METH$y > 3.4,
                         METH$y - 0.16,
                         METH$y + 0.16),
           colour = METH$col, linewidth = 0.45,
           linetype = "22") +
  annotate("text",
           x = METH$x, y = METH$y,
           label = METH$lab, colour = METH$col,
           size = 3.0, family = FONT,
           fontface = "bold") +

  ## stage labels beneath
  geom_text(data = STAGE,
            aes(x = x, y = 0.62, label = lab),
            size = 2.9, family = FONT,
            colour = "grey40", lineheight = 0.95) +

  labs(title = paste("Fusion Approaches Differ in Where",
                     "They Combine the Modalities")) +
  scale_x_continuous(limits = c(0.42, 5.62),
                     expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.15, 4.15),
                     expand = c(0, 0)) +
  theme(plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          family = FONT, margin = margin(b = 6)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

cat("\nwriting figure 3...\n")
print(system.time(
  save_fig(f, "fig3_fusion_schematic", 8.2, 4.4)))


install.packages("ggforce")

















# ---------------------------------------------------------
# fig3_fusion_schematic.R
# Where each fusion approach intervenes in the pipeline
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(ggforce)
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

theme_set(theme_void(base_size = 10,
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

BAND <- "#F4F7FA"
F_MOD <- "#DCE9F5"; S_MOD <- "#2E5C8A"
F_PRD <- "#FDEBD8"; S_PRD <- "#C4661F"
F_CMB <- "#FFE08A"; S_CMB <- "#B8860B"
F_SCR <- "#D8EFE6"; S_SCR <- "#1B7A5A"
F_CAL <- "#FFFFFF"; S_CAL <- "#7A7A7A"
TXT <- "#1A1A1A"

ORANGE <- "#C4661F"
PURPLE <- "#9B4F96"

## rounded-box helper: centre, half-width, half-height
rbox <- function(x, y, hw, hh, id) {
  tibble(id = id,
         x = c(x - hw, x + hw, x + hw, x - hw),
         y = c(y - hh, y - hh, y + hh, y + hh))
}

## ---- 2. layout ------------------------------------------

MOD <- c("EHR", "ECG", "CTPA")
YM <- c(3.15, 2.15, 1.15)

XR <- 1.05; XM <- 2.25; XP <- 3.35
XC <- 4.50; XS <- 5.60

modb <- bind_rows(
  lapply(seq_along(MOD), function(i)
    rbox(XR, YM[i], 0.42, 0.30, paste0("r", i))),
  lapply(seq_along(MOD), function(i)
    rbox(XM, YM[i], 0.46, 0.30, paste0("m", i))))

prdb <- bind_rows(
  lapply(seq_along(MOD), function(i)
    rbox(XP, YM[i], 0.30, 0.24, paste0("p", i))))

cmbb <- rbox(XC, 2.15, 0.44, 0.36, "c")
scrb <- rbox(XS, 2.15, 0.40, 0.30, "s")
calb <- rbox(XC, -0.30, 1.05, 0.60, "cal")

cat("\nmodality rows:", length(MOD), "\n")
cat("pipeline stages: 5\n")

## ---- 3. plot --------------------------------------------

f <- ggplot() +

  ## background band behind the pipeline
  annotate("rect", xmin = 0.45, xmax = 6.10,
           ymin = 0.70, ymax = 3.62,
           fill = BAND, colour = NA) +

  ## flow arrows
  annotate("segment",
           x = rep(c(XR + 0.44, XM + 0.48), each = 3),
           xend = rep(c(XM - 0.48, XP - 0.32), each = 3),
           y = rep(YM, 2), yend = rep(YM, 2),
           colour = "grey45", linewidth = 0.45,
           arrow = arrow(length = unit(0.075, "in"),
                         type = "closed")) +
  annotate("segment",
           x = rep(XP + 0.32, 3), xend = rep(XC - 0.46, 3),
           y = YM, yend = rep(2.15, 3),
           colour = "grey45", linewidth = 0.45,
           arrow = arrow(length = unit(0.075, "in"),
                         type = "closed")) +
  annotate("segment", x = XC + 0.46, xend = XS - 0.42,
           y = 2.15, yend = 2.15,
           colour = "grey45", linewidth = 0.45,
           arrow = arrow(length = unit(0.075, "in"),
                         type = "closed")) +

  ## boxes
  geom_shape(data = modb,
             aes(x = x, y = y, group = id),
             radius = unit(2.2, "mm"),
             fill = F_MOD, colour = S_MOD,
             linewidth = 0.5) +
  geom_shape(data = prdb,
             aes(x = x, y = y, group = id),
             radius = unit(2.2, "mm"),
             fill = F_PRD, colour = S_PRD,
             linewidth = 0.5) +
  geom_shape(data = cmbb,
             aes(x = x, y = y, group = id),
             radius = unit(2.6, "mm"),
             fill = F_CMB, colour = S_CMB,
             linewidth = 0.9) +
  geom_shape(data = scrb,
             aes(x = x, y = y, group = id),
             radius = unit(2.2, "mm"),
             fill = F_SCR, colour = S_SCR,
             linewidth = 0.5) +

  ## box labels
  annotate("text", x = XR, y = YM, label = MOD,
           size = 3.2, family = FONT, colour = TXT) +
  annotate("text", x = XM, y = YM, label = MOD,
           size = 3.2, family = FONT, colour = TXT) +
  annotate("text", x = XP, y = YM,
           label = paste0("p(", MOD, ")"),
           size = 2.9, family = FONT, colour = TXT) +
  annotate("text", x = XC, y = 2.15, label = "Combine",
           size = 3.3, family = FONT, colour = TXT,
           fontface = "bold") +
  annotate("text", x = XS, y = 2.15, label = "Fused\nscore",
           size = 3.1, family = FONT, colour = TXT,
           lineheight = 0.95) +

  ## interventions above the band
  annotate("segment", x = c(1.66, 2.82),
           xend = c(1.66, 2.82),
           y = c(3.62, 3.62), yend = c(4.02, 4.02),
           colour = c(ORANGE, PURPLE),
           linewidth = 0.5, linetype = "22") +
  annotate("text", x = c(1.66, 2.82), y = c(4.16, 4.16),
           label = c("Early fusion",
                     "Joint representation"),
           colour = c(ORANGE, PURPLE),
           size = 3.1, family = FONT,
           fontface = "bold") +

  ## callout below, holding the three combination rules
  annotate("segment", x = XC, xend = XC,
           y = 0.30, yend = 1.79,
           colour = S_CMB, linewidth = 0.5,
           linetype = "22") +
  geom_shape(data = calb,
             aes(x = x, y = y, group = id),
             radius = unit(2.4, "mm"),
             fill = F_CAL, colour = S_CAL,
             linewidth = 0.5) +
  annotate("text", x = XC, y = 0.10,
           label = "Combination rules evaluated",
           size = 2.95, family = FONT,
           fontface = "bold", colour = TXT) +
  annotate("text", x = XC,
           y = c(-0.24, -0.52, -0.78),
           label = c(
             "Weighted rank average (primary)",
             "Equal weighting; meta-learners",
             "Conditional gating"),
           size = 2.85, family = FONT, colour = "#3A3A3A") +

  ## stage labels
  annotate("text",
           x = c(XR, XM, XP, XC, XS), y = 0.50,
           label = c("Raw features", "Per-modality\nmodel",
                     "Modality\nprediction",
                     "Combination", "Output"),
           size = 2.85, family = FONT,
           colour = "grey35", lineheight = 0.95) +

  labs(title = paste("Fusion Approaches Differ in Where",
                     "They Combine the Modalities")) +
  scale_x_continuous(limits = c(0.30, 6.25),
                     expand = c(0, 0)) +
  scale_y_continuous(limits = c(-1.05, 4.45),
                     expand = c(0, 0)) +
  theme(plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          family = FONT, margin = margin(b = 4)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

cat("\nwriting figure 3...\n")
print(system.time(
  save_fig(f, "fig3_fusion_schematic", 8.6, 5.0)))
























# ---------------------------------------------------------
# fig21_time_to_event.R
# Distribution of days from admission to first event
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

BLUE <- "#0072B2"

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "decomp_p2_events.csv"),
              stringsAsFactors = FALSE)

cat("\nevents:", nrow(d), "\n")
cat(sprintf("mean %.2f  median %.1f  IQR %.0f-%.0f\n",
            mean(d$days), median(d$days),
            quantile(d$days, .25),
            quantile(d$days, .75)))
print(table(d$route))

mu <- mean(d$days)
md <- median(d$days)

f <- ggplot(d, aes(x = days)) +
  geom_vline(xintercept = c(7.5, 14.5),
             colour = "grey70", linewidth = 0.4,
             linetype = "22") +
  geom_histogram(binwidth = 1, boundary = 0,
                 fill = BLUE, colour = "white",
                 linewidth = 0.25) +
  geom_vline(xintercept = md, colour = "grey25",
             linewidth = 0.6) +
  annotate("text", x = md + 0.6, y = Inf,
           label = sprintf("median %.0f days", md),
           hjust = 0, vjust = 1.8, size = 3.0,
           colour = "grey25", family = FONT) +
  annotate("text", x = c(4, 11, 22),
           y = Inf, vjust = 3.6, size = 2.8,
           colour = "grey45", family = FONT,
           label = c("0-7 days", "8-14 days",
                     "15-30 days")) +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(
    mult = c(0, 0.16))) +
  labs(x = "Days From Admission to First Event",
       y = "Number of Admissions",
       title = paste("Composite Events Occur Throughout",
                     "the 30-Day Window")) +
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

cat("\nwriting figure 21...\n")
print(system.time(
  save_fig(f, "fig21_time_to_event", 7.6, 4.0)))























# ---------------------------------------------------------
# fig22_time_to_event_route.R
# Days to first event, split by ascertainment route
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

PAL <- c("All-Cause Death" = "#0072B2",
         "CV Readmission" = "#D55E00")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "decomp_p2_events.csv"),
              stringsAsFactors = FALSE)

pd <- d %>%
  mutate(route = ifelse(route == "Death",
                        "All-Cause Death",
                        "CV Readmission"),
         route = factor(route, levels = names(PAL)))

cat("\n--- by route ---\n")
print(pd %>% group_by(route) %>%
        summarise(n = n(), mean = mean(days),
                  median = median(days),
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

cat("\nroute by window:\n")
print(table(pd$route,
            cut(pd$days, c(-1, 7, 14, 30),
                labels = c("0-7d", "8-14d",
                           "15-30d"))))

lab <- pd %>%
  group_by(route) %>%
  summarise(n = n(), md = median(days),
            .groups = "drop") %>%
  mutate(txt = sprintf("%d events, median %.0f days",
                       n, md))

f <- ggplot(pd, aes(x = days, fill = route)) +
  geom_vline(xintercept = c(7.5, 14.5),
             colour = "grey70", linewidth = 0.4,
             linetype = "22") +
  geom_histogram(binwidth = 1, boundary = 0,
                 colour = "white", linewidth = 0.25) +
  geom_vline(data = lab,
             aes(xintercept = md, colour = route),
             linewidth = 0.6, show.legend = FALSE) +
  geom_text(data = lab,
            aes(x = 30, y = Inf, label = txt,
                colour = route),
            hjust = 1, vjust = 1.9, size = 3.0,
            family = FONT, show.legend = FALSE) +
  facet_wrap(~ route, ncol = 1) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(
    mult = c(0, 0.20))) +
  labs(x = "Days From Admission to First Event",
       y = "Number of Admissions",
       title = paste("Readmissions Occur Late;",
                     "Deaths Occur Throughout")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.y = unit(0.9, "lines"),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 22...\n")
print(system.time(
  save_fig(f, "fig22_time_to_event_route", 7.6, 5.4)))




























# ---------------------------------------------------------
# fig21_time_to_event.R
# Distribution of days from admission to first event
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

BLUE <- "#0072B2"

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "decomp_p2_events.csv"),
              stringsAsFactors = FALSE)

cat("\nevents:", nrow(d), "\n")
cat(sprintf("mean %.2f  median %.1f  IQR %.0f-%.0f\n",
            mean(d$days), median(d$days),
            quantile(d$days, .25),
            quantile(d$days, .75)))
print(table(d$route))

md <- median(d$days)
mu <- mean(d$days)
lab <- sprintf("%d events, median %.0f days",
               nrow(d), md)

## headroom above the tallest bar for the annotations
top <- max(table(cut(d$days, seq(-0.5, 30.5, 1))))
YMAX <- top * 1.30

f <- ggplot(d, aes(x = days)) +
  geom_vline(xintercept = c(7.5, 14.5),
             colour = "grey70", linewidth = 0.4,
             linetype = "22") +
  geom_histogram(binwidth = 1, boundary = 0,
                 fill = BLUE, colour = "white",
                 linewidth = 0.25) +
  geom_vline(xintercept = md, colour = BLUE,
             linewidth = 0.6) +
  ## summary top-right, matching Figure 22
  annotate("text", x = 30, y = YMAX * 0.965,
           label = lab, hjust = 1, vjust = 1,
           size = 3.0, colour = BLUE,
           family = FONT) +
  ## window labels on their own row, below the summary
  annotate("text", x = c(3.75, 11, 22.5),
           y = YMAX * 0.855, vjust = 1, size = 2.8,
           colour = "grey45", family = FONT,
           label = c("0-7 days", "8-14 days",
                     "15-30 days")) +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(0, YMAX),
    breaks = scales::pretty_breaks(5),
    expand = c(0, 0)) +
  labs(x = "Days From Admission to First Event",
       y = "Number of Admissions",
       title = paste("Composite Events Occur Throughout",
                     "the 30-Day Window")) +
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

cat("\nwriting figure 21...\n")
print(system.time(
  save_fig(f, "fig21_time_to_event", 7.6, 4.0)))



























# ---------------------------------------------------------
# fig20b_weights3cxr.R
# Fold weights in the three-modality model, CXR variant
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
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

PAL <- c("EHR" = "#0072B2", "ECG" = "#009E73",
         "CXR" = "#D55E00")
SHP <- c("EHR" = 16, "ECG" = 17, "CXR" = 15)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "wmean3cxr_weights_fixed.csv"),
              stringsAsFactors = FALSE)

pd <- d %>%
  pivot_longer(c("EHR", "ECG", "CXR"),
               names_to = "modality",
               values_to = "w") %>%
  mutate(outcome = recode(
           outcome,
           "Composite (30-day)" = "Composite (30-Day)",
           "30-day death" = "30-Day Death",
           "CV readmission" = "CV Readmission"),
         outcome = factor(outcome, levels = OUT_ORD),
         modality = factor(modality,
                           levels = names(PAL)))

cat("\n--- fold weights ---\n")
print(pd %>%
        group_by(outcome, modality) %>%
        summarise(min = min(w), max = max(w),
                  mean = mean(w), .groups = "drop") %>%
        as.data.frame(), digits = 3)

lab <- pd %>%
  filter(outcome == OUT_ORD[1], fold == 5) %>%
  mutate(txt = as.character(modality))

f <- ggplot(pd, aes(x = fold, y = w,
                    colour = modality,
                    shape = modality,
                    group = modality)) +
  geom_line(linewidth = 0.6, alpha = 0.65) +
  geom_point(size = 2.7) +
  geom_text(data = lab,
            aes(label = txt), hjust = -0.35,
            size = 3.0, fontface = "bold",
            family = FONT, show.legend = FALSE) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_x_continuous(breaks = 1:5,
                     limits = c(0.7, 5.9),
                     expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(-0.03, 1.03),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  labs(x = "Cross-Validation Fold",
       y = "Fold-Selected Weight",
       title = paste("The Weight Grid Retains the CXR",
                     "Modality in Every Fold")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.x = unit(0.9, "lines"),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 9),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 20b...\n")
print(system.time(
  save_fig(f, "fig20b_weights3cxr", 8.6, 4.0)))

































# ---------------------------------------------------------
# fig20b_weights3cxr.R
# Fusion weight split, CXR as the third modality
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
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

PAL <- c("EHR" = "#0072B2", "ECG" = "#E69F00",
         "CXR" = "#CC79A7")
TXT <- c("EHR" = "#00558A", "ECG" = "#8A5E00",
         "CXR" = "#8A4A72")

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")
MOD <- c("EHR", "ECG", "CXR")

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "wmean3cxr_weights_fixed.csv"),
              stringsAsFactors = FALSE)

lg <- d %>%
  pivot_longer(all_of(MOD), names_to = "modality",
               values_to = "w") %>%
  mutate(outcome = recode(
           outcome,
           "Composite (30-day)" = "Composite (30-Day)",
           "30-day death" = "30-Day Death",
           "CV readmission" = "CV Readmission"),
         outcome = factor(outcome,
                          levels = rev(OUT_ORD)),
         modality = factor(modality, levels = MOD))

sm <- lg %>%
  group_by(outcome, modality) %>%
  summarise(mean = mean(w), lo = min(w),
            hi = max(w), .groups = "drop")

cat("\n--- mean weights and fold ranges ---\n")
print(as.data.frame(sm), digits = 3)

pd <- sm %>%
  arrange(outcome, modality) %>%
  group_by(outcome) %>%
  mutate(hi_c = cumsum(mean),
         lo_c = hi_c - mean,
         mid = (lo_c + hi_c) / 2) %>%
  ungroup() %>%
  mutate(yy = as.numeric(outcome))

## fold-range annotation, one line per outcome
ann <- sm %>%
  arrange(outcome, modality) %>%
  group_by(outcome) %>%
  summarise(txt = paste0(
    "CXR ", sprintf("%.2f", lo[modality == "CXR"]),
    "\u2013", sprintf("%.2f", hi[modality == "CXR"]),
    "; ECG ", sprintf("%.2f", lo[modality == "ECG"]),
    "\u2013", sprintf("%.2f", hi[modality == "ECG"])),
    .groups = "drop") %>%
  mutate(yy = as.numeric(outcome))

HH <- 0.30
N_RULE <- c("EHR" = 0, "ECG" = 7, "CXR" = 4)

hatch <- do.call(rbind, lapply(
  seq_len(nrow(pd)), function(i) {
    r <- pd[i, ]
    k <- N_RULE[[as.character(r$modality)]]
    if (k < 1) return(NULL)
    xs <- seq(r$lo_c, r$hi_c,
              length.out = k + 2)
    xs <- xs[c(-1, -length(xs))]
    data.frame(x = xs, yy = r$yy)
  }))

lab_top <- pd %>% filter(yy == max(yy))

## ---- 3. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.2, 0.8, by = 0.2),
             colour = "grey93", linewidth = 0.3) +
  geom_rect(data = pd,
            aes(xmin = lo_c, xmax = hi_c,
                ymin = yy - HH, ymax = yy + HH,
                fill = modality),
            colour = "white", linewidth = 0.5) +
  geom_segment(data = hatch,
               aes(x = x, xend = x,
                   y = yy - HH + 0.015,
                   yend = yy + HH - 0.015),
               colour = "white", linewidth = 0.35,
               inherit.aes = FALSE) +
  geom_label(data = pd,
             aes(x = mid, y = yy,
                 label = sprintf("%.2f", mean)),
             size = 3.0, family = FONT,
             label.size = 0.25, label.r = unit(1, "mm"),
             fill = "white", colour = "grey15") +
  geom_text(data = lab_top,
            aes(x = mid, y = yy + HH + 0.30,
                label = modality, colour = modality),
            fontface = "bold", size = 3.2,
            family = FONT, show.legend = FALSE) +
  geom_text(data = ann,
            aes(x = 1.02, y = yy, label = txt),
            hjust = 0, size = 2.75, colour = "grey35",
            family = FONT) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = TXT, guide = "none") +
  scale_x_continuous(
    limits = c(0, 1.34),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(OUT_ORD),
    labels = rev(OUT_ORD),
    limits = c(0.45, length(OUT_ORD) + 0.9),
    expand = c(0, 0)) +
  labs(x = "Share of the Fusion Weight (Sums to 1.0)",
       y = NULL,
       title = paste("Fusion Weight Split With CXR as",
                     "the Third Modality")) +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 9.5),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(
          size = 9, margin = margin(t = 8)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 20b...\n")
print(system.time(
  save_fig(f, "fig20b_weights3cxr", 9.0, 3.6)))



















# ---------------------------------------------------------
# fig1_cause_of_death.R
# Cause of inpatient death after acute pulmonary embolism
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

ORANGE <- "#D55E00"   # Okabe-Ito vermillion
GREY   <- "#7F7F7F"
DARK   <- "#333333"

## ---- 2. data --------------------------------------------
## Percentages and counts as reported by Earle et al.
## (2023); N = 131 inpatient deaths among 2,052 admissions.

dat <- tribble(
  ~cause,               ~pct, ~n,
  "Cancer-related",      32,  41,
  "Pulmonary embolism",  22,  29,
  "Infection or sepsis", 17,  22,
  "Other",               15,  20,
  "Stroke",               7,   9,
  "Pulmonary",            7,   9
)

dat <- dat %>%
  mutate(
    cause = factor(cause, levels = rev(cause)),
    key   = cause %in% c("Cancer-related",
                         "Pulmonary embolism"),
    lab   = paste0(pct, "%  (n = ", n, ")")
  )

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = pct, y = cause)) +
  geom_col(aes(fill = key), width = 0.68) +
  geom_text(aes(label = lab, colour = key),
            hjust = -0.08, size = 3.1,
            family = FONT, fontface = "bold") +
  scale_fill_manual(values = c("TRUE"  = ORANGE,
                               "FALSE" = GREY),
                    guide = "none") +
  scale_colour_manual(values = c("TRUE"  = ORANGE,
                                 "FALSE" = DARK),
                      guide = "none") +
  scale_x_continuous(
    limits = c(0, 42),
    breaks = seq(0, 40, 10),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = paste0("Cancer Is the Leading Cause of ",
                   "Inpatient Death After Pulmonary ",
                   "Embolism"),
    x = "Share of inpatient deaths",
    y = NULL
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 8)),
    axis.title.x = element_text(size = 9.5,
                                colour = DARK,
                                margin = margin(t = 6)),
    axis.text.y = element_text(size = 10, colour = DARK),
    axis.text.x = element_text(size = 9, colour = GREY),
    panel.grid.major.x = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 10, 4, 4)
  )

save_fig(p, "fig1_cause_of_death", width = 6.4,
         height = 3.0)
































# ---------------------------------------------------------
# fig1_cause_of_death.R
# Cause of inpatient death after acute pulmonary embolism,
# stratified by PE severity
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
GREEN  <- "#009E73"
ORANGE <- "#E69F00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. data --------------------------------------------
## Totals are as reported in Earle et al. (2023) text.
## Segment values are read off the published figure, so
## they are approximate.

dat <- tribble(
  ~cause,                ~severity,      ~pct,
  "Cancer-related",      "Low",          16.2,
  "Cancer-related",      "Intermediate", 15.0,
  "Cancer-related",      "High",          0.8,
  "Pulmonary embolism",  "Low",           0.0,
  "Pulmonary embolism",  "Intermediate",  4.5,
  "Pulmonary embolism",  "High",         17.9,
  "Infection or sepsis", "Low",           4.6,
  "Infection or sepsis", "Intermediate",  9.2,
  "Infection or sepsis", "High",          3.1,
  "Other",               "Low",           4.6,
  "Other",               "Intermediate",  8.4,
  "Other",               "High",          2.3,
  "Stroke",              "Low",           3.1,
  "Stroke",              "Intermediate",  3.8,
  "Stroke",              "High",          0.0,
  "Pulmonary",           "Low",           0.8,
  "Pulmonary",           "Intermediate",  5.4,
  "Pulmonary",           "High",          0.8
)

ORD <- dat %>%
  group_by(cause) %>%
  summarise(total = sum(pct), .groups = "drop") %>%
  arrange(total)

dat <- dat %>%
  mutate(
    cause = factor(cause, levels = ORD$cause),
    severity = factor(severity,
                      levels = c("Low", "Intermediate",
                                 "High"))
  )

TOT <- ORD %>%
  mutate(cause = factor(cause, levels = ORD$cause),
         lab = paste0(round(total), "%"))

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = pct, y = cause)) +
  geom_col_pattern(
    aes(fill = severity,
        pattern = severity,
        pattern_angle = severity),
    width = 0.68,
    colour = "white",
    linewidth = 0.4,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_density = 0.12,
    pattern_spacing = 0.020,
    pattern_key_scale_factor = 0.7
  ) +
  geom_text(data = TOT,
            aes(x = total, y = cause, label = lab),
            hjust = -0.25, size = 3.1,
            family = FONT, fontface = "bold",
            colour = DARK, inherit.aes = FALSE) +
  scale_fill_manual(
    values = c("Low" = BLUE,
               "Intermediate" = GREEN,
               "High" = ORANGE),
    name = "PE severity"
  ) +
  scale_pattern_manual(
    values = c("Low" = "none",
               "Intermediate" = "stripe",
               "High" = "crosshatch"),
    name = "PE severity"
  ) +
  scale_pattern_angle_manual(
    values = c("Low" = 0,
               "Intermediate" = 45,
               "High" = 135),
    name = "PE severity"
  ) +
  scale_x_continuous(
    limits = c(0, 38),
    breaks = seq(0, 35, 5),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = paste0("Cancer Is the Leading Cause of ",
                   "Inpatient Death After Pulmonary ",
                   "Embolism"),
    x = "Share of inpatient deaths",
    y = NULL
  ) +
  guides(
    fill = guide_legend(nrow = 1, byrow = TRUE),
    pattern = guide_legend(nrow = 1, byrow = TRUE),
    pattern_angle = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 8)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 6)),
    axis.text.y = element_text(size = 10, colour = DARK),
    axis.text.x = element_text(size = 9, colour = GREY),
    panel.grid.major.x = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.title = element_text(size = 9, colour = DARK),
    legend.text = element_text(size = 9, colour = DARK),
    legend.margin = margin(0, 0, 2, 0),
    legend.key.size = unit(0.42, "cm"),
    plot.margin = margin(6, 10, 4, 4)
  )

save_fig(p, "fig1_cause_of_death", width = 6.6,
         height = 3.4)


























# ---------------------------------------------------------
# fig1_cause_of_death.R
# Cause of inpatient death after acute pulmonary embolism,
# stratified by PE severity
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
GREEN  <- "#009E73"
ORANGE <- "#E69F00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. data --------------------------------------------
## Totals are as reported in Earle et al. (2023) text.
## Segment values are read off the published figure, so
## they are approximate.

dat <- tribble(
  ~cause,                ~severity,      ~pct,
  "Cancer-related",      "Low",          16.2,
  "Cancer-related",      "Intermediate", 15.0,
  "Cancer-related",      "High",          0.8,
  "Pulmonary embolism",  "Low",           0.0,
  "Pulmonary embolism",  "Intermediate",  4.5,
  "Pulmonary embolism",  "High",         17.9,
  "Infection or sepsis", "Low",           4.6,
  "Infection or sepsis", "Intermediate",  9.2,
  "Infection or sepsis", "High",          3.1,
  "Other",               "Low",           4.6,
  "Other",               "Intermediate",  8.4,
  "Other",               "High",          2.3,
  "Stroke",              "Low",           3.1,
  "Stroke",              "Intermediate",  3.8,
  "Stroke",              "High",          0.0,
  "Pulmonary",           "Low",           0.8,
  "Pulmonary",           "Intermediate",  5.4,
  "Pulmonary",           "High",          0.8
)

ORD <- dat %>%
  group_by(cause) %>%
  summarise(total = sum(pct), .groups = "drop") %>%
  arrange(total)

dat <- dat %>%
  mutate(
    cause = factor(cause, levels = ORD$cause),
    severity = factor(severity,
                      levels = c("Low", "Intermediate",
                                 "High"))
  )

TOT <- ORD %>%
  mutate(cause = factor(cause, levels = ORD$cause),
         lab = paste0(round(total), "%"))

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = pct, y = cause)) +
  geom_col_pattern(
    aes(fill = severity,
        pattern = severity,
        pattern_angle = severity),
    position = position_stack(reverse = TRUE),
    width = 0.68,
    colour = "white",
    linewidth = 0.4,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_density = 0.12,
    pattern_spacing = 0.020,
    pattern_key_scale_factor = 0.7
  ) +
  geom_text(data = TOT,
            aes(x = total, y = cause, label = lab),
            hjust = -0.25, size = 3.1,
            family = FONT, fontface = "bold",
            colour = DARK, inherit.aes = FALSE) +
  scale_fill_manual(
    values = c("Low" = BLUE,
               "Intermediate" = GREEN,
               "High" = ORANGE),
    name = "PE severity at presentation:"
  ) +
  scale_pattern_manual(
    values = c("Low" = "none",
               "Intermediate" = "stripe",
               "High" = "crosshatch"),
    name = "PE severity at presentation:"
  ) +
  scale_pattern_angle_manual(
    values = c("Low" = 0,
               "Intermediate" = 45,
               "High" = 135),
    name = "PE severity at presentation:"
  ) +
  scale_x_continuous(
    limits = c(0, 38),
    breaks = seq(0, 35, 5),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = paste0("Cancer Is the Leading Cause of ",
                   "Inpatient Death After Pulmonary ",
                   "Embolism"),
    x = "Share of inpatient deaths",
    y = NULL
  ) +
  guides(
    fill = guide_legend(nrow = 1, byrow = TRUE,
                        label.position = "right"),
    pattern = guide_legend(nrow = 1, byrow = TRUE,
                           label.position = "right"),
    pattern_angle = guide_legend(nrow = 1, byrow = TRUE,
                                 label.position = "right")
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 6)),
    axis.text.y = element_text(size = 10, colour = DARK),
    axis.text.x = element_text(size = 9, colour = GREY),
    panel.grid.major.x = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.direction = "horizontal",
    legend.title = element_text(
      size = 9, colour = DARK,
      margin = margin(r = 12)),
    legend.text = element_text(
      size = 9, colour = DARK,
      margin = margin(l = 4, r = 22)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    legend.box.spacing = unit(0.15, "cm"),
    plot.margin = margin(6, 12, 4, 4)
  )

save_fig(p, "fig1_cause_of_death", width = 6.8,
         height = 3.5)




























# ---------------------------------------------------------
# fig2_cause_by_survival.R
# How the cause of death shifts with time since PE
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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
GREEN  <- "#009E73"
YELLOW <- "#E69F00"
GREY   <- "#7F7F7F"
DARK   <- "#333333"

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded; the per-window percentages are not given in the
## text, so the values are approximate.
##
## Ten original categories collapsed to five:
##   PE-related      = PE or associated complications
##                     + recurrent PE
##   Other or
##   undetermined    = pulmonary, bleeding, suicide,
##                     other, not determinable

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         ">3 years\n(n = 106)")

dat <- tribble(
  ~cause,                 ~w, ~pct,
  "PE-related",            1,   71,
  "PE-related",            2,    5,
  "PE-related",            3,    6,
  "PE-related",            4,    2,
  "Cancer",                1,   16,
  "Cancer",                2,   56,
  "Cancer",                3,   32,
  "Cancer",                4,   11,
  "Infection",             1,    9,
  "Infection",             2,   13,
  "Infection",             3,   12,
  "Infection",             4,   20,
  "Cardiovascular",        1,    2,
  "Cardiovascular",        2,    8,
  "Cardiovascular",        3,   14,
  "Cardiovascular",        4,   19,
  "Other or undetermined", 1,    2,
  "Other or undetermined", 2,   18,
  "Other or undetermined", 3,   36,
  "Other or undetermined", 4,   48
)

LEV <- c("PE-related", "Cancer", "Other or undetermined",
         "Infection", "Cardiovascular")

dat <- dat %>%
  mutate(cause = factor(cause, levels = LEV))

ENDS <- dat %>% filter(w == 4)

COL <- c("PE-related"            = BLUE,
         "Cancer"                = ORANGE,
         "Other or undetermined" = GREY,
         "Infection"             = YELLOW,
         "Cardiovascular"        = GREEN)

SHP <- c("PE-related"            = 16,
         "Cancer"                = 17,
         "Other or undetermined" = 4,
         "Infection"             = 15,
         "Cardiovascular"        = 18)

LTY <- c("PE-related"            = "solid",
         "Cancer"                = "22",
         "Other or undetermined" = "11",
         "Infection"             = "4212",
         "Cardiovascular"        = "solid")

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = w, y = pct,
                     colour = cause,
                     shape = cause,
                     linetype = cause)) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.4) +
  geom_text(data = ENDS,
            aes(label = cause),
            hjust = 0, nudge_x = 0.10,
            size = 3.1, fontface = "bold",
            family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = COL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY, guide = "none") +
  scale_x_continuous(
    breaks = 1:4, labels = WIN,
    limits = c(0.85, 6.15),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 78),
    breaks = seq(0, 75, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  labs(
    title = paste0("The Embolism Kills Early; Cancer, ",
                   "Infection and Cardiovascular Disease ",
                   "Kill Later"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths in window"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 8)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 7.0,
         height = 3.6)























# ---------------------------------------------------------
# fig2_cause_by_survival.R
# How the cause of death shifts with time since PE
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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
GREEN  <- "#009E73"
YELLOW <- "#E69F00"
GREY   <- "#7F7F7F"
DARK   <- "#333333"

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded; the per-window percentages are not given in the
## text, so the values are approximate.
##
## Ten original categories collapsed to five:
##   PE-related      = PE or associated complications
##                     + recurrent PE
##   Other or
##   undetermined    = pulmonary, bleeding, suicide,
##                     other, not determinable

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         ">3 years\n(n = 106)")

dat <- tribble(
  ~cause,                 ~w, ~pct,
  "PE-related",            1,   71,
  "PE-related",            2,    5,
  "PE-related",            3,    6,
  "PE-related",            4,    2,
  "Cancer",                1,   16,
  "Cancer",                2,   56,
  "Cancer",                3,   32,
  "Cancer",                4,   11,
  "Infection",             1,    9,
  "Infection",             2,   13,
  "Infection",             3,   12,
  "Infection",             4,   20,
  "Cardiovascular",        1,    2,
  "Cardiovascular",        2,    8,
  "Cardiovascular",        3,   14,
  "Cardiovascular",        4,   19,
  "Other or undetermined", 1,    2,
  "Other or undetermined", 2,   18,
  "Other or undetermined", 3,   36,
  "Other or undetermined", 4,   48
)

LEV <- c("PE-related", "Cancer", "Other or undetermined",
         "Infection", "Cardiovascular")

dat <- dat %>%
  mutate(cause = factor(cause, levels = LEV))

ENDS <- tribble(
  ~cause,                 ~ypos,
  "PE-related",             2,
  "Cancer",                11,
  "Other or undetermined", 48,
  "Infection",             24,
  "Cardiovascular",        15
) %>%
  mutate(cause = factor(cause, levels = LEV), w = 4)

COL <- c("PE-related"            = BLUE,
         "Cancer"                = ORANGE,
         "Other or undetermined" = GREY,
         "Infection"             = YELLOW,
         "Cardiovascular"        = GREEN)

SHP <- c("PE-related"            = 16,
         "Cancer"                = 17,
         "Other or undetermined" = 4,
         "Infection"             = 15,
         "Cardiovascular"        = 18)

LTY <- c("PE-related"            = "solid",
         "Cancer"                = "22",
         "Other or undetermined" = "11",
         "Infection"             = "4212",
         "Cardiovascular"        = "solid")

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = w, y = pct,
                     colour = cause,
                     shape = cause,
                     linetype = cause)) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.4) +
  geom_text(data = ENDS,
            aes(x = w, y = ypos, label = cause,
                colour = cause),
            hjust = 0, nudge_x = 0.12,
            size = 3.1, fontface = "bold",
            family = FONT,
            inherit.aes = FALSE,
            show.legend = FALSE) +
  scale_colour_manual(values = COL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY, guide = "none") +
  scale_x_continuous(
    breaks = 1:4, labels = WIN,
    limits = c(0.85, 6.15),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 78),
    breaks = seq(0, 75, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  labs(
    title = paste0("The Embolism Kills Early; Cancer, ",
                   "Infection and Cardiovascular Disease ",
                   "Kill Later"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths in window"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                hjust = 0.31,
                                margin = margin(t = 8)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 7.0,
         height = 4.0)






































# ---------------------------------------------------------
# fig2_cause_by_survival.R
# How the cause of death shifts with time since PE
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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
GREEN  <- "#009E73"
YELLOW <- "#E69F00"
GREY   <- "#7F7F7F"
DARK   <- "#333333"

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded; the per-window percentages are not given in the
## text, so the values are approximate.
##
## Ten original categories collapsed to five:
##   PE-related      = PE or associated complications
##                     + recurrent PE
##   Other or
##   undetermined    = pulmonary, bleeding, suicide,
##                     other, not determinable

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         ">3 years\n(n = 106)")

dat <- tribble(
  ~cause,          ~w, ~pct,
  "PE-related",     1,   71,
  "PE-related",     2,    5,
  "PE-related",     3,    6,
  "PE-related",     4,    2,
  "Cancer",         1,   16,
  "Cancer",         2,   56,
  "Cancer",         3,   32,
  "Cancer",         4,   11,
  "Infection",      1,    9,
  "Infection",      2,   13,
  "Infection",      3,   12,
  "Infection",      4,   20,
  "Cardiovascular", 1,    2,
  "Cardiovascular", 2,    8,
  "Cardiovascular", 3,   14,
  "Cardiovascular", 4,   19,
  "Other",          1,    2,
  "Other",          2,   18,
  "Other",          3,   36,
  "Other",          4,   48
)

LEV <- c("PE-related", "Cancer", "Other",
         "Infection", "Cardiovascular")

dat <- dat %>%
  mutate(cause = factor(cause, levels = LEV))

ENDS <- tribble(
  ~cause,           ~ypos, ~lab,
  "PE-related",        2,  "PE-related",
  "Cancer",           11,  "Cancer",
  "Other",            48,  "Other or\nundetermined",
  "Infection",        23,  "Infection",
  "Cardiovascular",   16,  "Cardiovascular"
) %>%
  mutate(cause = factor(cause, levels = LEV), w = 4)

COL <- c("PE-related"     = BLUE,
         "Cancer"         = ORANGE,
         "Other"          = GREY,
         "Infection"      = YELLOW,
         "Cardiovascular" = GREEN)

SHP <- c("PE-related"     = 16,
         "Cancer"         = 17,
         "Other"          = 4,
         "Infection"      = 15,
         "Cardiovascular" = 18)

LTY <- c("PE-related"     = "solid",
         "Cancer"         = "22",
         "Other"          = "11",
         "Infection"      = "4212",
         "Cardiovascular" = "solid")

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = w, y = pct,
                     colour = cause,
                     shape = cause,
                     linetype = cause)) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.4) +
  geom_text(data = ENDS,
            aes(x = w, y = ypos, label = lab,
                colour = cause),
            hjust = 0, nudge_x = 0.07,
            size = 3.0, fontface = "bold",
            lineheight = 0.92,
            family = FONT,
            inherit.aes = FALSE,
            show.legend = FALSE) +
  scale_colour_manual(values = COL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY, guide = "none") +
  scale_x_continuous(
    breaks = 1:4, labels = WIN,
    limits = c(0.88, 4.92),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 76),
    breaks = seq(0, 75, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  coord_cartesian(clip = "off") +
  labs(
    title = paste0("The Embolism Kills Early; Cancer, ",
                   "Infection and Cardiovascular Disease ",
                   "Kill Later"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths in window"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                hjust = 0.38,
                                margin = margin(t = 8)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 4, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 6.6,
         height = 3.9)
































# ---------------------------------------------------------
# fig2_cause_by_survival.R
# How the cause of death shifts with time since PE
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
GREEN  <- "#009E73"
YELLOW <- "#E69F00"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded; the per-window percentages are not given in the
## text, so the values are approximate.
##
## Ten original categories collapsed to five:
##   PE-related      = PE or associated complications
##                     + recurrent PE
##   Other or
##   undetermined    = pulmonary, bleeding, suicide,
##                     other, not determinable

WIN <- c("0-30 days (n = 78)",
         "31-365 days (n = 97)",
         "1-3 years (n = 80)",
         "Over 3 years (n = 106)")

CAUSE <- c("Other or undetermined", "Cardiovascular",
           "Infection", "Cancer", "PE-related")

dat <- tribble(
  ~cause,                  ~win, ~pct,
  "PE-related",             1,     71,
  "PE-related",             2,      5,
  "PE-related",             3,      6,
  "PE-related",             4,      2,
  "Cancer",                 1,     16,
  "Cancer",                 2,     56,
  "Cancer",                 3,     32,
  "Cancer",                 4,     11,
  "Infection",              1,      9,
  "Infection",              2,     13,
  "Infection",              3,     12,
  "Infection",              4,     20,
  "Cardiovascular",         1,      2,
  "Cardiovascular",         2,      8,
  "Cardiovascular",         3,     14,
  "Cardiovascular",         4,     19,
  "Other or undetermined",  1,      2,
  "Other or undetermined",  2,     18,
  "Other or undetermined",  3,     36,
  "Other or undetermined",  4,     48
) %>%
  mutate(
    cause = factor(cause, levels = CAUSE),
    win   = factor(WIN[win], levels = WIN)
  )

FILLS <- setNames(c(BLUE, ORANGE, YELLOW, GREEN), WIN)
PATS  <- setNames(c("none", "stripe", "crosshatch",
                    "circle"), WIN)
ANGS  <- setNames(c(0, 45, 135, 0), WIN)

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = pct, y = cause)) +
  geom_col_pattern(
    aes(fill = win, pattern = win,
        pattern_angle = win),
    position = position_dodge2(reverse = TRUE,
                               padding = 0.12),
    width = 0.78,
    colour = "white",
    linewidth = 0.3,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_density = 0.11,
    pattern_spacing = 0.016,
    pattern_key_scale_factor = 0.6
  ) +
  scale_fill_manual(values = FILLS, name = NULL) +
  scale_pattern_manual(values = PATS, name = NULL) +
  scale_pattern_angle_manual(values = ANGS,
                             name = NULL) +
  scale_x_continuous(
    limits = c(0, 76),
    breaks = seq(0, 75, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("The Embolism Kills Early; Cancer, ",
                   "Infection and Cardiovascular Disease ",
                   "Kill Later"),
    x = "Share of deaths occurring within the window",
    y = NULL
  ) +
  guides(
    fill = guide_legend(nrow = 1, byrow = TRUE),
    pattern = guide_legend(nrow = 1, byrow = TRUE),
    pattern_angle = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 8)),
    axis.text.y = element_text(size = 10, colour = DARK),
    axis.text.x = element_text(size = 9, colour = GREY),
    panel.grid.major.x = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.direction = "horizontal",
    legend.text = element_text(size = 8.8, colour = DARK,
                               margin = margin(l = 3,
                                               r = 14)),
    legend.key.size = unit(0.38, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    legend.box.spacing = unit(0.1, "cm"),
    plot.margin = margin(6, 8, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 7.0,
         height = 4.2)




















# ---------------------------------------------------------
# fig2_cause_by_survival.R
# Causes of death by time from PE diagnosis
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
SKY    <- "#56B4E9"
ORANGE <- "#D55E00"
YELLOW <- "#E69F00"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
GREY   <- "#999999"
DARK   <- "#333333"

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded to sum to 100 within each window; the per-window
## percentages are not given in the paper text, so the
## values are approximate.

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         "Over 3 years\n(n = 106)")

CAUSE <- c("PE or associated complications",
           "Recurrent PE",
           "Cancer",
           "Infections",
           "Cardiovascular events",
           "Pulmonary causes",
           "Bleeding (including intracranial)",
           "Suicide",
           "Other",
           "Not determinable")

dat <- tribble(
  ~cause, ~w, ~pct,
  "PE or associated complications",     1, 67,
  "PE or associated complications",     2,  1,
  "PE or associated complications",     3,  0,
  "PE or associated complications",     4,  0,
  "Recurrent PE",                       1,  4,
  "Recurrent PE",                       2,  4,
  "Recurrent PE",                       3,  6,
  "Recurrent PE",                       4,  2,
  "Cancer",                             1, 16,
  "Cancer",                             2, 56,
  "Cancer",                             3, 32,
  "Cancer",                             4, 11,
  "Infections",                         1,  9,
  "Infections",                         2, 13,
  "Infections",                         3, 12,
  "Infections",                         4, 20,
  "Cardiovascular events",              1,  2,
  "Cardiovascular events",              2,  8,
  "Cardiovascular events",              3, 14,
  "Cardiovascular events",              4, 19,
  "Pulmonary causes",                   1,  0,
  "Pulmonary causes",                   2,  3,
  "Pulmonary causes",                   3,  4,
  "Pulmonary causes",                   4,  4,
  "Bleeding (including intracranial)",  1,  1,
  "Bleeding (including intracranial)",  2,  2,
  "Bleeding (including intracranial)",  3,  3,
  "Bleeding (including intracranial)",  4,  4,
  "Suicide",                            1,  0,
  "Suicide",                            2,  2,
  "Suicide",                            3,  0,
  "Suicide",                            4,  0,
  "Other",                              1,  1,
  "Other",                              2,  3,
  "Other",                              3,  3,
  "Other",                              4,  9,
  "Not determinable",                   1,  0,
  "Not determinable",                   2,  8,
  "Not determinable",                   3, 26,
  "Not determinable",                   4, 31
) %>%
  mutate(
    cause = factor(cause, levels = CAUSE),
    win   = factor(WIN[w], levels = WIN)
  )

FILLS <- setNames(
  c(BLUE, SKY, ORANGE, YELLOW, GREEN,
    GREEN, PURPLE, PURPLE, SKY, GREY),
  CAUSE)

PATS <- setNames(
  c("none", "stripe", "none", "stripe", "none",
    "crosshatch", "none", "stripe", "circle",
    "crosshatch"),
  CAUSE)

ANGS <- setNames(
  c(0, 45, 0, 135, 0,
    45, 0, 45, 0, 0),
  CAUSE)

SPACE <- setNames(
  c(0.02, 0.030, 0.02, 0.018, 0.02,
    0.024, 0.02, 0.014, 0.030, 0.020),
  CAUSE)

DENS <- setNames(
  c(0.10, 0.16, 0.10, 0.12, 0.10,
    0.10, 0.10, 0.20, 0.30, 0.14),
  CAUSE)

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = win, y = pct)) +
  geom_col_pattern(
    aes(fill = cause,
        pattern = cause,
        pattern_angle = cause,
        pattern_spacing = cause,
        pattern_density = cause),
    position = position_stack(reverse = TRUE),
    width = 0.62,
    colour = "white",
    linewidth = 0.35,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_key_scale_factor = 0.5
  ) +
  scale_fill_manual(values = FILLS, name = NULL) +
  scale_pattern_manual(values = PATS, name = NULL) +
  scale_pattern_angle_manual(values = ANGS,
                             name = NULL) +
  scale_pattern_spacing_manual(values = SPACE,
                               name = NULL) +
  scale_pattern_density_manual(values = DENS,
                               name = NULL) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("The Embolism Kills Early; Cancer, ",
                   "Infection and Cardiovascular Disease ",
                   "Kill Later"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths within the window"
  ) +
  guides(
    fill = guide_legend(ncol = 2, byrow = FALSE),
    pattern = guide_legend(ncol = 2, byrow = FALSE),
    pattern_angle = guide_legend(ncol = 2,
                                 byrow = FALSE),
    pattern_spacing = guide_legend(ncol = 2,
                                   byrow = FALSE),
    pattern_density = guide_legend(ncol = 2,
                                   byrow = FALSE)
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 8)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.justification = "top",
    legend.text = element_text(size = 8.6,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 8)),
    legend.key.size = unit(0.40, "cm"),
    legend.spacing.y = unit(0.05, "cm"),
    plot.margin = margin(6, 4, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 8.0,
         height = 4.4)





























# ---------------------------------------------------------
# fig2_cause_by_survival.R
# Causes of death by time from PE diagnosis
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
SKY    <- "#56B4E9"
ORANGE <- "#D55E00"
YELLOW <- "#E69F00"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
GREY   <- "#999999"
DARK   <- "#333333"

## Only segments at or above this size get a value label.
LABEL_MIN <- 7

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded to sum to 100 within each window; the per-window
## percentages are not given in the paper text, so the
## values are approximate.

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         "Over 3 years\n(n = 106)")

CAUSE <- c("PE or associated complications",
           "Recurrent PE",
           "Cancer",
           "Infections",
           "Cardiovascular events",
           "Pulmonary causes",
           "Bleeding (including intracranial)",
           "Suicide",
           "Other",
           "Not determinable")

dat <- tribble(
  ~cause, ~w, ~pct,
  "PE or associated complications",     1, 67,
  "PE or associated complications",     2,  1,
  "PE or associated complications",     3,  0,
  "PE or associated complications",     4,  0,
  "Recurrent PE",                       1,  4,
  "Recurrent PE",                       2,  4,
  "Recurrent PE",                       3,  6,
  "Recurrent PE",                       4,  2,
  "Cancer",                             1, 16,
  "Cancer",                             2, 56,
  "Cancer",                             3, 32,
  "Cancer",                             4, 11,
  "Infections",                         1,  9,
  "Infections",                         2, 13,
  "Infections",                         3, 12,
  "Infections",                         4, 20,
  "Cardiovascular events",              1,  2,
  "Cardiovascular events",              2,  8,
  "Cardiovascular events",              3, 14,
  "Cardiovascular events",              4, 19,
  "Pulmonary causes",                   1,  0,
  "Pulmonary causes",                   2,  3,
  "Pulmonary causes",                   3,  4,
  "Pulmonary causes",                   4,  4,
  "Bleeding (including intracranial)",  1,  1,
  "Bleeding (including intracranial)",  2,  2,
  "Bleeding (including intracranial)",  3,  3,
  "Bleeding (including intracranial)",  4,  4,
  "Suicide",                            1,  0,
  "Suicide",                            2,  2,
  "Suicide",                            3,  0,
  "Suicide",                            4,  0,
  "Other",                              1,  1,
  "Other",                              2,  3,
  "Other",                              3,  3,
  "Other",                              4,  9,
  "Not determinable",                   1,  0,
  "Not determinable",                   2,  8,
  "Not determinable",                   3, 26,
  "Not determinable",                   4, 31
) %>%
  mutate(
    cause = factor(cause, levels = CAUSE),
    win   = factor(WIN[w], levels = WIN)
  )

LABS <- dat %>%
  arrange(win, cause) %>%
  group_by(win) %>%
  mutate(top = cumsum(pct),
         mid = top - pct / 2) %>%
  ungroup() %>%
  filter(pct >= LABEL_MIN) %>%
  mutate(lab = paste0(pct, "%"))

FILLS <- setNames(
  c(BLUE, SKY, ORANGE, YELLOW, GREEN,
    GREEN, PURPLE, PURPLE, SKY, GREY),
  CAUSE)

PATS <- setNames(
  c("none", "stripe", "none", "stripe", "none",
    "crosshatch", "none", "stripe", "circle",
    "crosshatch"),
  CAUSE)

ANGS <- setNames(
  c(0, 45, 0, 135, 0,
    45, 0, 45, 0, 0),
  CAUSE)

SPACE <- setNames(
  c(0.02, 0.030, 0.02, 0.018, 0.02,
    0.024, 0.02, 0.014, 0.030, 0.020),
  CAUSE)

DENS <- setNames(
  c(0.10, 0.16, 0.10, 0.12, 0.10,
    0.10, 0.10, 0.20, 0.30, 0.14),
  CAUSE)

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = win, y = pct)) +
  geom_col_pattern(
    aes(fill = cause,
        pattern = cause,
        pattern_angle = cause,
        pattern_spacing = cause,
        pattern_density = cause),
    position = position_stack(reverse = TRUE),
    width = 0.62,
    colour = "white",
    linewidth = 0.35,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_key_scale_factor = 0.5
  ) +
  geom_label(
    data = LABS,
    aes(x = win, y = mid, label = lab),
    inherit.aes = FALSE,
    fill = "white",
    colour = DARK,
    alpha = 0.88,
    label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.9,
    fontface = "bold",
    family = FONT
  ) +
  scale_fill_manual(values = FILLS, name = NULL) +
  scale_pattern_manual(values = PATS, name = NULL) +
  scale_pattern_angle_manual(values = ANGS,
                             name = NULL) +
  scale_pattern_spacing_manual(values = SPACE,
                               name = NULL) +
  scale_pattern_density_manual(values = DENS,
                               name = NULL) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("The Embolism Kills Early; Cancer, ",
                   "Infection and Cardiovascular Disease ",
                   "Kill Later"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths within the window"
  ) +
  guides(
    fill = guide_legend(ncol = 1),
    pattern = guide_legend(ncol = 1),
    pattern_angle = guide_legend(ncol = 1),
    pattern_spacing = guide_legend(ncol = 1),
    pattern_density = guide_legend(ncol = 1)
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 8)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.justification = "top",
    legend.text = element_text(size = 8.6,
                               colour = DARK,
                               margin = margin(l = 3)),
    legend.key.size = unit(0.38, "cm"),
    legend.spacing.y = unit(0.02, "cm"),
    legend.margin = margin(2, 0, 0, 2),
    plot.margin = margin(6, 4, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 6.9,
         height = 4.6)













# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

hal <- function(mapping, data) {
  geom_label(data = data, mapping = mapping,
             inherit.aes = FALSE,
             fill = "white", colour = DARK,
             alpha = 0.90, label.size = 0,
             label.r = unit(0.06, "lines"),
             label.padding = unit(0.10, "lines"),
             size = 2.7, fontface = "bold",
             family = FONT)
}

## =========================================================
## FIGURE 12a - image content against acquisition context
## =========================================================

## Values typed in from an earlier run that predates the
## CXR aggregation fix. Later versions of this figure
## read them from cxr_metadata_ci.csv.
END <- c("Composite", "30-day death",
         "In-hospital death", "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,           ~source,            ~auc,  ~lo,   ~hi,
  "Composite",         "CXR image model",  0.658, 0.615, 0.700,
  "Composite",         "Film portability", 0.587, NA,    NA,
  "Composite",         "Total film count", 0.618, NA,    NA,
  "30-day death",      "CXR image model",  0.672, 0.628, 0.713,
  "30-day death",      "Film portability", 0.631, NA,    NA,
  "30-day death",      "Total film count", 0.638, NA,    NA,
  "In-hospital death", "CXR image model",  0.658, 0.612, 0.703,
  "In-hospital death", "Film portability", 0.674, NA,    NA,
  "In-hospital death", "Total film count", 0.711, NA,    NA,
  "CV readmission",    "CXR image model",  0.624, 0.548, 0.697,
  "CV readmission",    "Film portability", 0.469, NA,    NA,
  "CV readmission",    "Total film count", 0.549, NA,    NA
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

pa <- ggplot(a, aes(x = endpoint, y = auc)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = position_dodge(width = 0.78),
    width = 0.70,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi, group = source),
    position = position_dodge(width = 0.78),
    width = 0.16, linewidth = 0.45,
    colour = DARK, na.rm = TRUE
  ) +
  hal(aes(x = endpoint, y = auc, label = lab,
          group = source), a) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(
    limits = c(0.40, 0.78),
    breaks = seq(0.40, 0.75, 0.05),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("Acquisition Context Rivals the Image ",
                   "for Mortality but Not for Readmission"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.8, height = 3.8)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,  ~n,
  "Composite",    "All portable",  0.631, 431,
  "Composite",    "Mixed",         0.643, 167,
  "Composite",    "None portable", 0.646, 319,
  "30-day death", "All portable",  0.653, 431,
  "30-day death", "Mixed",         0.649, 167,
  "30-day death", "None portable", 0.627, 319
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0("n = ", n))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

pb <- ggplot(b, aes(x = endpoint, y = auc)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = position_dodge(width = 0.74),
    width = 0.66,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  hal(aes(x = endpoint, y = auc, label = lab,
          group = stratum), b) +
  geom_text(aes(y = auc + 0.016, label = nlab,
                group = stratum),
            position = position_dodge(width = 0.74),
            size = 2.5, colour = GREY, family = FONT) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(
    limits = c(0.40, 0.72),
    breaks = seq(0.40, 0.70, 0.05),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("Discrimination Is Flat Across ",
                   "Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 5.6, height = 3.6)































# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## =========================================================
## FIGURE 12a - image content against acquisition context
## =========================================================

END <- c("Composite", "30-day death",
         "In-hospital death", "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,           ~source,            ~auc,  ~lo,   ~hi,
  "Composite",         "CXR image model",  0.658, 0.615, 0.700,
  "Composite",         "Film portability", 0.587, NA,    NA,
  "Composite",         "Total film count", 0.618, NA,    NA,
  "30-day death",      "CXR image model",  0.672, 0.628, 0.713,
  "30-day death",      "Film portability", 0.631, NA,    NA,
  "30-day death",      "Total film count", 0.638, NA,    NA,
  "In-hospital death", "CXR image model",  0.658, 0.612, 0.703,
  "In-hospital death", "Film portability", 0.674, NA,    NA,
  "In-hospital death", "Total film count", 0.711, NA,    NA,
  "CV readmission",    "CXR image model",  0.624, 0.548, 0.697,
  "CV readmission",    "Film portability", 0.469, NA,    NA,
  "CV readmission",    "Total film count", 0.549, NA,    NA
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK, na.rm = TRUE
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.75)) +
  labs(
    title = paste0("Acquisition Context Rivals the Image ",
                   "for Mortality but Not for Readmission"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.8, height = 3.8)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,  ~n,
  "Composite",    "All portable",  0.631, 431,
  "Composite",    "Mixed",         0.643, 167,
  "Composite",    "None portable", 0.646, 319,
  "30-day death", "All portable",  0.653, 431,
  "30-day death", "Mixed",         0.649, 167,
  "30-day death", "None portable", 0.627, 319
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0("n = ", n))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(label = nlab),
    position = DODGE_B,
    vjust = -0.6,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.70, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.70)) +
  labs(
    title = paste0("Discrimination Is Flat Across ",
                   "Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 5.8, height = 3.6)


































# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.75)) +
  labs(
    title = paste0("The Image Model Outperforms ",
                   "Acquisition Context on Every ",
                   "Endpoint"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 3.8)

## =========================================================
## FIGURE 12b - discrimination within portability strata
##
## These values are typed in rather than read from a
## CSV, and predate the CXR aggregation fix. Later
## versions of this figure read cxr_strata_ci.csv.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,  ~n,
  "Composite",    "All portable",  0.631, 431,
  "Composite",    "Mixed",         0.643, 167,
  "Composite",    "None portable", 0.646, 319,
  "30-day death", "All portable",  0.653, 431,
  "30-day death", "Mixed",         0.649, 167,
  "30-day death", "None portable", 0.627, 319
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0("n = ", n))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(label = nlab),
    position = DODGE_B,
    vjust = -0.6,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.70, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.70)) +
  labs(
    title = paste0("Discrimination Is Flat Across ",
                   "Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 5.8, height = 3.6)



































# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.75)) +
  labs(
    title = paste0("The Image Model Outperforms ",
                   "Acquisition Context on Every ",
                   "Endpoint"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 3.8)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## Source: data/cxr_strata_ci.csv
##
## Cardiovascular readmission is excluded: 19, 15 and 20
## events per stratum give intervals too wide to interpret.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,   ~lo,    ~hi,    ~n,  ~ev,
  "Composite",    "All portable",  0.6303, 0.5747, 0.6845, 468, 119,
  "Composite",    "Mixed",         0.6462, 0.5585, 0.7333, 184,  47,
  "Composite",    "None portable", 0.6817, 0.5995, 0.7680, 375,  44,
  "30-day death", "All portable",  0.6456, 0.5865, 0.7007, 468, 103,
  "30-day death", "Mixed",         0.6441, 0.5415, 0.7470, 184,  32,
  "30-day death", "None portable", 0.6788, 0.5586, 0.7888, 375,  26
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0(ev, " ev"))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_B,
    width = 0.14, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(y = hi, label = nlab),
    position = DODGE_B,
    vjust = -0.7,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.85, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.85)) +
  labs(
    title = paste0("No Portability Effect Is Detectable, ",
                   "but the Strata Cannot Rule One Out"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5,
                              hjust = 0,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 5.8, height = 3.8)





























# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------

nice <- c(
  age = "Age",
  hematocrit = "Haematocrit",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  sodium = "Sodium",
  chloride = "Chloride",
  bicarbonate = "Bicarbonate",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  heart_rate = "Heart rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  cancer = "Malignancy",
  heart_failure = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title,
                           n_feat = 15) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = n_feat)

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  # thin for legibility on large cohorts
  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE, size = 0.38,
                     alpha = 0.38, width = 0.34) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5)) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      barheight = unit(2.6, "cm"),
      barwidth = unit(0.34, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 11.5,
                                hjust = 0,
                                margin = margin(b = 8)),
      axis.title.x = element_text(size = 9.5,
                                  colour = DARK,
                                  margin = margin(t = 6)),
      axis.text.y = element_text(size = 9.5,
                                 colour = DARK),
      axis.text.x = element_text(size = 9,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 8.6,
                                  colour = DARK),
      legend.text = element_text(size = 8.6,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 4),
      plot.margin = margin(6, 4, 4, 4))
}

## ---- 4. build and save ----------------------------------

a <- beeswarm_panel(
  "death_30d",
  "Frailty and Disease Burden Dominate 30-Day Mortality")
save_fig(a, "fig7a_beeswarm_death_30d",
         width = 7.0, height = 4.8)

b <- beeswarm_panel(
  "composite_30d",
  "Cardiac Comorbidity Enters for the Composite Endpoint")
save_fig(b, "fig7b_beeswarm_composite_30d",
         width = 7.0, height = 4.8)

c <- beeswarm_panel(
  "cv_first",
  "Heart Failure Leads for Cardiovascular Readmission")
save_fig(c, "fig7c_beeswarm_cv_first",
         width = 7.0, height = 4.8)























# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

theme_set(theme_minimal(base_size = 11,
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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------
## Short forms added after seeing raw names in the output.

nice <- c(
  age = "Age",
  hct = "Haematocrit",
  hematocrit = "Haematocrit",
  hgb = "Haemoglobin",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  plt = "Platelets",
  sodium = "Sodium",
  na = "Sodium",
  chloride = "Chloride",
  cl = "Chloride",
  bicarb = "Bicarbonate",
  bicarbonate = "Bicarbonate",
  aniongap = "Anion gap",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  creat = "Creatinine",
  hr = "Heart rate",
  heart_rate = "Heart rate",
  rr = "Respiratory rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  temp = "Temperature",
  wbc = "Leukocytes",
  glucose = "Glucose",
  potassium = "Potassium",
  k = "Potassium",
  calcium = "Calcium",
  magnesium = "Magnesium",
  phosphate = "Phosphate",
  albumin = "Albumin",
  cancer = "Malignancy",
  malignancy = "Malignancy",
  heart_failure = "Heart failure",
  chf = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  afib = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title,
                           n_feat = 15) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = n_feat)

  # report any name still falling through
  miss <- setdiff(ord$feature, names(nice))
  if (length(miss) > 0) {
    message("no label for: ",
            paste(miss, collapse = ", "))
  }

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE, size = 0.55,
                     alpha = 0.42, width = 0.38) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5),
      guide = guide_colourbar(reverse = TRUE)) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      reverse = TRUE,
      barheight = unit(3.4, "cm"),
      barwidth = unit(0.40, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 13,
                                hjust = 0,
                                margin = margin(b = 10)),
      axis.title.x = element_text(size = 11,
                                  colour = DARK,
                                  margin = margin(t = 8)),
      axis.text.y = element_text(size = 11,
                                 colour = DARK),
      axis.text.x = element_text(size = 10,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 9.5,
                                  colour = DARK),
      legend.text = element_text(size = 9.5,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 6),
      plot.margin = margin(8, 6, 6, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 9.0
HT <- 6.4

a <- beeswarm_panel(
  "death_30d",
  "Frailty and Disease Burden Dominate 30-Day Mortality")
save_fig(a, "fig7a_beeswarm_death_30d",
         width = WD, height = HT)

b <- beeswarm_panel(
  "composite_30d",
  "Cardiac Comorbidity Enters for the Composite Endpoint")
save_fig(b, "fig7b_beeswarm_composite_30d",
         width = WD, height = HT)

c <- beeswarm_panel(
  "cv_first",
  "Heart Failure Leads for Cardiovascular Readmission")
save_fig(c, "fig7c_beeswarm_cv_first",
         width = WD, height = HT)





































  
# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

theme_set(theme_minimal(base_size = 11,
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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------

nice <- c(
  age = "Age",
  hct = "Haematocrit",
  hematocrit = "Haematocrit",
  hgb = "Haemoglobin",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  plt = "Platelets",
  sodium = "Sodium",
  na = "Sodium",
  chloride = "Chloride",
  cl = "Chloride",
  bicarb = "Bicarbonate",
  bicarbonate = "Bicarbonate",
  aniongap = "Anion gap",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  creat = "Creatinine",
  hr = "Heart rate",
  heart_rate = "Heart rate",
  rr = "Respiratory rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  temp = "Temperature",
  wbc = "Leukocytes",
  glucose = "Glucose",
  potassium = "Potassium",
  k = "Potassium",
  calcium = "Calcium",
  magnesium = "Magnesium",
  phosphate = "Phosphate",
  albumin = "Albumin",
  cancer = "Malignancy",
  malignancy = "Malignancy",
  heart_failure = "Heart failure",
  chf = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  afib = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title,
                           n_feat = 15) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = n_feat)

  miss <- setdiff(ord$feature, names(nice))
  if (length(miss) > 0) {
    message("no label for: ",
            paste(miss, collapse = ", "))
  }

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE, size = 0.55,
                     alpha = 0.42, width = 0.38) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5)) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      reverse = TRUE,
      barheight = unit(3.4, "cm"),
      barwidth = unit(0.40, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 13,
                                hjust = 0.5,
                                margin = margin(b = 10)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 11,
                                  colour = DARK,
                                  margin = margin(t = 8)),
      axis.text.y = element_text(size = 11,
                                 colour = DARK),
      axis.text.x = element_text(size = 10,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 9.5,
                                  colour = DARK),
      legend.text = element_text(size = 9.5,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 6),
      plot.margin = margin(8, 6, 6, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 9.0
HT <- 6.4

a <- beeswarm_panel(
  "death_30d",
  paste0("Feature Contributions to Predicted ",
         "30-Day Mortality"))
save_fig(a, "fig7a_beeswarm_death_30d",
         width = WD, height = HT)

b <- beeswarm_panel(
  "composite_30d",
  paste0("Feature Contributions to the Predicted ",
         "30-Day Composite Endpoint"))
save_fig(b, "fig7b_beeswarm_composite_30d",
         width = WD, height = HT)

c <- beeswarm_panel(
  "cv_first",
  paste0("Feature Contributions to Predicted ",
         "Cardiovascular Readmission"))
save_fig(c, "fig7c_beeswarm_cv_first",
         width = WD, height = HT)
































# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## ---- 2. shared theme ------------------------------------

house <- function() {
  theme(
    plot.title = element_text(face = "bold",
                              size = 12,
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 9.5,
                                colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )
}

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.75)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Against Acquisition Context"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 3.8)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## Source: data/cxr_strata_ci.csv
##
## Cardiovascular readmission is excluded: 19, 15 and 20
## events per stratum give intervals too wide to interpret.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,   ~lo,    ~hi,    ~ev,
  "Composite",    "All portable",  0.6303, 0.5747, 0.6845, 119,
  "Composite",    "Mixed",         0.6462, 0.5585, 0.7333,  47,
  "Composite",    "None portable", 0.6817, 0.5995, 0.7680,  44,
  "30-day death", "All portable",  0.6456, 0.5865, 0.7007, 103,
  "30-day death", "Mixed",         0.6441, 0.5415, 0.7470,  32,
  "30-day death", "None portable", 0.6788, 0.5586, 0.7888,  26
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0(ev, " ev"))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_B,
    width = 0.14, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(y = hi, label = nlab),
    position = DODGE_B,
    vjust = -0.7,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.85, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.85)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Within Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 5.8, height = 3.8)
































# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

theme_set(theme_minimal(base_size = 11,
                        base_family = FONT))

WRITE_PDF <- FALSE

SPLIT <- FALSE

SWARM_W <- 0.78
PT_SIZE <- 0.72
PT_ALPHA <- 0.45
N_FEAT <- 15

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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------

nice <- c(
  age = "Age",
  hct = "Haematocrit",
  hematocrit = "Haematocrit",
  hgb = "Haemoglobin",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  plt = "Platelets",
  sodium = "Sodium",
  na = "Sodium",
  chloride = "Chloride",
  cl = "Chloride",
  bicarb = "Bicarbonate",
  bicarbonate = "Bicarbonate",
  aniongap = "Anion gap",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  creat = "Creatinine",
  hr = "Heart rate",
  heart_rate = "Heart rate",
  rr = "Respiratory rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  temp = "Temperature",
  wbc = "Leukocytes",
  glucose = "Glucose",
  potassium = "Potassium",
  k = "Potassium",
  calcium = "Calcium",
  magnesium = "Magnesium",
  phosphate = "Phosphate",
  albumin = "Albumin",
  cancer = "Malignancy",
  malignancy = "Malignancy",
  heart_failure = "Heart failure",
  chf = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  afib = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title,
                           rank_from = 1,
                           rank_to = N_FEAT) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = N_FEAT) %>%
    slice(rank_from:rank_to)

  miss <- setdiff(ord$feature, names(nice))
  if (length(miss) > 0) {
    message("no label for: ",
            paste(miss, collapse = ", "))
  }

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE,
                     size = PT_SIZE,
                     alpha = PT_ALPHA,
                     width = SWARM_W) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5)) +
    scale_y_discrete(expand = expansion(
      add = c(0.6, 0.6))) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      reverse = TRUE,
      barheight = unit(3.8, "cm"),
      barwidth = unit(0.42, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 13,
                                hjust = 0.5,
                                margin = margin(b = 12)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 11,
                                  colour = DARK,
                                  margin = margin(t = 8)),
      axis.text.y = element_text(size = 11.5,
                                 colour = DARK),
      axis.text.x = element_text(size = 10,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 9.5,
                                  colour = DARK),
      legend.text = element_text(size = 9.5,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 6),
      plot.margin = margin(8, 6, 6, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 9.0

ROW_H <- 0.62
PAD <- 1.6

JOBS <- list(
  list(oc = "death_30d",
       nm = "fig7a_beeswarm_death_30d",
       ti = paste0("Feature Contributions to Predicted ",
                   "30-Day Mortality")),
  list(oc = "composite_30d",
       nm = "fig7b_beeswarm_composite_30d",
       ti = paste0("Feature Contributions to the ",
                   "Predicted 30-Day Composite ",
                   "Endpoint")),
  list(oc = "cv_first",
       nm = "fig7c_beeswarm_cv_first",
       ti = paste0("Feature Contributions to Predicted ",
                   "Cardiovascular Readmission"))
)

for (j in JOBS) {
  if (!SPLIT) {
    p <- beeswarm_panel(j$oc, j$ti, 1, N_FEAT)
    save_fig(p, j$nm, width = WD,
             height = N_FEAT * ROW_H + PAD)
  } else {
    p1 <- beeswarm_panel(
      j$oc, paste0(j$ti, ": Ranks 1 to 8"), 1, 8)
    save_fig(p1, paste0(j$nm, "_i"),
             width = WD, height = 8 * ROW_H + PAD)

    p2 <- beeswarm_panel(
      j$oc, paste0(j$ti, ": Ranks 9 to 15"), 9, 15)
    save_fig(p2, paste0(j$nm, "_ii"),
             width = WD, height = 7 * ROW_H + PAD)
  }
}







































# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

theme_set(theme_minimal(base_size = 11,
                        base_family = FONT))

WRITE_PDF <- FALSE

## ---- display settings -----------------------------------

N_FEAT  <- 12
ROW_H   <- 0.85
SWARM_W <- 0.55
X_Q     <- 0.995

PT_SIZE  <- 0.80
PT_ALPHA <- 0.45
PAD_H    <- 1.8

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

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------

nice <- c(
  age = "Age",
  hct = "Haematocrit",
  hematocrit = "Haematocrit",
  hgb = "Haemoglobin",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  plt = "Platelets",
  sodium = "Sodium",
  na = "Sodium",
  chloride = "Chloride",
  cl = "Chloride",
  bicarb = "Bicarbonate",
  bicarbonate = "Bicarbonate",
  aniongap = "Anion gap",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  creat = "Creatinine",
  hr = "Heart rate",
  heart_rate = "Heart rate",
  rr = "Respiratory rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  temp = "Temperature",
  wbc = "Leukocytes",
  glucose = "Glucose",
  potassium = "Potassium",
  k = "Potassium",
  calcium = "Calcium",
  magnesium = "Magnesium",
  phosphate = "Phosphate",
  albumin = "Albumin",
  cancer = "Malignancy",
  malignancy = "Malignancy",
  heart_failure = "Heart failure",
  chf = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  afib = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = N_FEAT)

  miss <- setdiff(ord$feature, names(nice))
  if (length(miss) > 0) {
    message("no label for: ",
            paste(miss, collapse = ", "))
  }

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  xl <- as.numeric(
    quantile(abs(d$shap), X_Q, na.rm = TRUE))
  xl <- ceiling(xl * 10) / 10
  n_out <- sum(abs(d$shap) > xl, na.rm = TRUE)
  message(outcome, ": x limit +/-", xl,
          "  points off-panel: ", n_out,
          " of ", nrow(d))

  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE,
                     size = PT_SIZE,
                     alpha = PT_ALPHA,
                     width = SWARM_W) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5)) +
    scale_y_discrete(expand = expansion(
      add = c(0.7, 0.7))) +
    coord_cartesian(xlim = c(-xl, xl)) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      reverse = TRUE,
      barheight = unit(4.2, "cm"),
      barwidth = unit(0.45, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 14,
                                hjust = 0.5,
                                margin = margin(b = 14)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 10)),
      axis.text.y = element_text(size = 12.5,
                                 colour = DARK),
      axis.text.x = element_text(size = 11,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 10.5,
                                  colour = DARK),
      legend.text = element_text(size = 10.5,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 8),
      plot.margin = margin(10, 6, 8, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 9.0
HT <- N_FEAT * ROW_H + PAD_H

JOBS <- list(
  list(oc = "death_30d",
       nm = "fig7a_beeswarm_death_30d",
       ti = paste0("Feature Contributions to Predicted ",
                   "30-Day Mortality")),
  list(oc = "composite_30d",
       nm = "fig7b_beeswarm_composite_30d",
       ti = paste0("Feature Contributions to the ",
                   "Predicted 30-Day Composite ",
                   "Endpoint")),
  list(oc = "cv_first",
       nm = "fig7c_beeswarm_cv_first",
       ti = paste0("Feature Contributions to Predicted ",
                   "Cardiovascular Readmission"))
)

for (j in JOBS) {
  p <- beeswarm_panel(j$oc, j$ti)
  save_fig(p, j$nm, width = WD, height = HT)
}





































# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

theme_set(theme_minimal(base_size = 12,
                        base_family = FONT))

WRITE_PDF <- FALSE

## Set TRUE to split each endpoint into two figures.
SPLIT <- FALSE

## ---- geometry -------------------------------------------

ROW_H <- 1.15
PAD <- 2.0
SWARM_W <- 0.42
PT_SIZE <- 0.85
PT_ALPHA <- 0.42
N_FEAT <- 15
WD <- 12.5

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white",
         limitsize = FALSE)
  message("saved: ", p_png,
          "  (", width, " x ", height, " in)")
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white",
           family = "sans", limitsize = FALSE)
    message("saved: ", p_pdf)
  }
}

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------

nice <- c(
  age = "Age",
  hct = "Haematocrit",
  hematocrit = "Haematocrit",
  hgb = "Haemoglobin",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  plt = "Platelets",
  sodium = "Sodium",
  na = "Sodium",
  chloride = "Chloride",
  cl = "Chloride",
  bicarb = "Bicarbonate",
  bicarbonate = "Bicarbonate",
  aniongap = "Anion gap",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  creat = "Creatinine",
  hr = "Heart rate",
  heart_rate = "Heart rate",
  rr = "Respiratory rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  temp = "Temperature",
  wbc = "Leukocytes",
  glucose = "Glucose",
  potassium = "Potassium",
  k = "Potassium",
  calcium = "Calcium",
  magnesium = "Magnesium",
  phosphate = "Phosphate",
  albumin = "Albumin",
  cancer = "Malignancy",
  malignancy = "Malignancy",
  heart_failure = "Heart failure",
  chf = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  afib = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title,
                           rank_from = 1,
                           rank_to = N_FEAT) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = N_FEAT) %>%
    slice(rank_from:rank_to)

  message("features plotted: ", nrow(ord))

  miss <- setdiff(ord$feature, names(nice))
  if (length(miss) > 0) {
    message("no label for: ",
            paste(miss, collapse = ", "))
  }

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE,
                     size = PT_SIZE,
                     alpha = PT_ALPHA,
                     width = SWARM_W) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5)) +
    scale_y_discrete(expand = expansion(
      add = c(0.55, 0.55))) +
    scale_x_continuous(
      expand = expansion(mult = c(0.02, 0.02))) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      reverse = TRUE,
      barheight = unit(4.5, "cm"),
      barwidth = unit(0.45, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 15,
                                hjust = 0.5,
                                margin = margin(b = 14)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 10)),
      axis.text.y = element_text(size = 13,
                                 colour = DARK),
      axis.text.x = element_text(size = 11,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 11,
                                  colour = DARK),
      legend.text = element_text(size = 11,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 8),
      plot.margin = margin(10, 8, 8, 4))
}

## ---- 4. build and save ----------------------------------

JOBS <- list(
  list(oc = "death_30d",
       nm = "fig7a_beeswarm_death_30d",
       ti = paste0("Feature Contributions to Predicted ",
                   "30-Day Mortality")),
  list(oc = "composite_30d",
       nm = "fig7b_beeswarm_composite_30d",
       ti = paste0("Feature Contributions to the ",
                   "Predicted 30-Day Composite ",
                   "Endpoint")),
  list(oc = "cv_first",
       nm = "fig7c_beeswarm_cv_first",
       ti = paste0("Feature Contributions to Predicted ",
                   "Cardiovascular Readmission"))
)

for (j in JOBS) {
  if (!SPLIT) {
    p <- beeswarm_panel(j$oc, j$ti, 1, N_FEAT)
    save_fig(p, j$nm, width = WD,
             height = N_FEAT * ROW_H + PAD)
  } else {
    p1 <- beeswarm_panel(
      j$oc, paste0(j$ti, ": Ranks 1 to 8"), 1, 8)
    save_fig(p1, paste0(j$nm, "_i"),
             width = WD, height = 8 * ROW_H + PAD)

    p2 <- beeswarm_panel(
      j$oc, paste0(j$ti, ": Ranks 9 to 15"), 9, 15)
    save_fig(p2, paste0(j$nm, "_ii"),
             width = WD, height = 7 * ROW_H + PAD)
  }
}




























# ---------------------------------------------------------
# fig7_ehr_beeswarm.R
# Per-patient feature contributions for the EHR modality
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

theme_set(theme_minimal(base_size = 12,
                        base_family = FONT))

WRITE_PDF <- FALSE

## Set TRUE to split each endpoint into two figures.
SPLIT <- FALSE

## ---- geometry -------------------------------------------

ROW_H <- 1.15
PAD <- 2.0
SWARM_W <- 0.42
PT_SIZE <- 0.85
PT_ALPHA <- 0.42
N_FEAT <- 15
WD <- 9.5

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white",
         limitsize = FALSE)
  message("saved: ", p_png,
          "  (", width, " x ", height, " in)")
  if (WRITE_PDF) {
    p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
    ggsave(p_pdf, plot, device = pdf,
           width = width, height = height,
           units = "in", bg = "white",
           family = "sans", limitsize = FALSE)
    message("saved: ", p_pdf)
  }
}

## Okabe-Ito
BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. readable feature labels -------------------------

nice <- c(
  age = "Age",
  hct = "Haematocrit",
  hematocrit = "Haematocrit",
  hgb = "Haemoglobin",
  hemoglobin = "Haemoglobin",
  mch = "MCH",
  mchc = "MCHC",
  mcv = "MCV",
  rbc = "Erythrocytes",
  platelets = "Platelets",
  plt = "Platelets",
  sodium = "Sodium",
  na = "Sodium",
  chloride = "Chloride",
  cl = "Chloride",
  bicarb = "Bicarbonate",
  bicarbonate = "Bicarbonate",
  aniongap = "Anion gap",
  anion_gap = "Anion gap",
  bun = "Urea nitrogen",
  creatinine = "Creatinine",
  creat = "Creatinine",
  hr = "Heart rate",
  heart_rate = "Heart rate",
  rr = "Respiratory rate",
  resp_rate = "Respiratory rate",
  sbp = "Systolic BP",
  dbp = "Diastolic BP",
  spo2 = "Oxygen saturation",
  temperature = "Temperature",
  temp = "Temperature",
  wbc = "Leukocytes",
  glucose = "Glucose",
  potassium = "Potassium",
  k = "Potassium",
  calcium = "Calcium",
  magnesium = "Magnesium",
  phosphate = "Phosphate",
  albumin = "Albumin",
  cancer = "Malignancy",
  malignancy = "Malignancy",
  heart_failure = "Heart failure",
  chf = "Heart failure",
  atrial_fibrillation = "Atrial fibrillation",
  afib = "Atrial fibrillation",
  copd = "COPD"
)

## ---- 3. panel builder -----------------------------------

beeswarm_panel <- function(outcome, title,
                           rank_from = 1,
                           rank_to = N_FEAT) {
  d <- read.csv(
    file.path(
      data_dir,
      paste0("shap_long_", outcome, "_target.csv")),
    stringsAsFactors = FALSE)

  ord <- d %>%
    group_by(feature) %>%
    summarise(m = mean(abs(shap)),
              .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = N_FEAT) %>%
    slice(rank_from:rank_to)

  message("features plotted: ", nrow(ord))

  miss <- setdiff(ord$feature, names(nice))
  if (length(miss) > 0) {
    message("no label for: ",
            paste(miss, collapse = ", "))
  }

  labs_ord <- ifelse(ord$feature %in% names(nice),
                     nice[ord$feature],
                     ord$feature)

  d <- d %>%
    filter(feature %in% ord$feature) %>%
    mutate(
      label = ifelse(feature %in% names(nice),
                     nice[feature], feature),
      label = factor(label,
                     levels = rev(labs_ord)),
      zc = pmax(pmin(zval, 2.5), -2.5))

  if (nrow(d) > 40000) {
    set.seed(42)
    d <- d %>%
      group_by(label) %>%
      slice_sample(n = 2500) %>%
      ungroup()
  }

  ggplot(d, aes(x = shap, y = label, colour = zc)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_quasirandom(groupOnX = FALSE,
                     size = PT_SIZE,
                     alpha = PT_ALPHA,
                     width = SWARM_W) +
    scale_colour_gradient2(
      low = BLUE, mid = "grey90",
      high = ORANGE, midpoint = 0,
      name = "Feature value\n(SD from source mean)",
      breaks = c(-2, 0, 2),
      labels = c("Low", "Average", "High"),
      limits = c(-2.5, 2.5)) +
    scale_y_discrete(expand = expansion(
      add = c(0.55, 0.55))) +
    scale_x_continuous(
      expand = expansion(mult = c(0.02, 0.02))) +
    labs(
      x = paste0("Contribution to predicted risk ",
                 "(log-odds)"),
      y = NULL,
      title = title) +
    guides(colour = guide_colourbar(
      reverse = TRUE,
      barheight = unit(4.5, "cm"),
      barwidth = unit(0.45, "cm"),
      ticks = FALSE)) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 15,
                                hjust = 0.5,
                                margin = margin(b = 14)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 10)),
      axis.text.y = element_text(size = 13,
                                 colour = DARK),
      axis.text.x = element_text(size = 11,
                                 colour = GREY),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "right",
      legend.title = element_text(size = 11,
                                  colour = DARK),
      legend.text = element_text(size = 11,
                                 colour = DARK),
      legend.margin = margin(0, 0, 0, 8),
      plot.margin = margin(10, 8, 8, 4))
}

## ---- 4. build and save ----------------------------------

JOBS <- list(
  list(oc = "death_30d",
       nm = "fig7a_beeswarm_death_30d",
       ti = paste0("Feature Contributions to Predicted ",
                   "30-Day Mortality")),
  list(oc = "composite_30d",
       nm = "fig7b_beeswarm_composite_30d",
       ti = paste0("Feature Contributions to the ",
                   "Predicted 30-Day Composite ",
                   "Endpoint")),
  list(oc = "cv_first",
       nm = "fig7c_beeswarm_cv_first",
       ti = paste0("Feature Contributions to Predicted ",
                   "Cardiovascular Readmission"))
)

for (j in JOBS) {
  if (!SPLIT) {
    p <- beeswarm_panel(j$oc, j$ti, 1, N_FEAT)
    save_fig(p, j$nm, width = WD,
             height = N_FEAT * ROW_H + PAD)
  } else {
    p1 <- beeswarm_panel(
      j$oc, paste0(j$ti, ": Ranks 1 to 8"), 1, 8)
    save_fig(p1, paste0(j$nm, "_i"),
             width = WD, height = 8 * ROW_H + PAD)

    p2 <- beeswarm_panel(
      j$oc, paste0(j$ti, ": Ranks 9 to 15"), 9, 15)
    save_fig(p2, paste0(j$nm, "_ii"),
             width = WD, height = 7 * ROW_H + PAD)
  }
}




























# ---------------------------------------------------------
# fig1_cause_of_death.R
# Cause of inpatient death after acute pulmonary embolism,
# stratified by PE severity
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
GREEN  <- "#009E73"
ORANGE <- "#E69F00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

## ---- 2. data --------------------------------------------
## Totals are as reported in Earle et al. (2023) text.
## Segment values are read off the published figure, so
## they are approximate.

dat <- tribble(
  ~cause,                ~severity,      ~pct,
  "Cancer-related",      "Low",          16.2,
  "Cancer-related",      "Intermediate", 15.0,
  "Cancer-related",      "High",          0.8,
  "Pulmonary embolism",  "Low",           0.0,
  "Pulmonary embolism",  "Intermediate",  4.5,
  "Pulmonary embolism",  "High",         17.9,
  "Infection or sepsis", "Low",           4.6,
  "Infection or sepsis", "Intermediate",  9.2,
  "Infection or sepsis", "High",          3.1,
  "Other",               "Low",           4.6,
  "Other",               "Intermediate",  8.4,
  "Other",               "High",          2.3,
  "Stroke",              "Low",           3.1,
  "Stroke",              "Intermediate",  3.8,
  "Stroke",              "High",          0.0,
  "Pulmonary",           "Low",           0.8,
  "Pulmonary",           "Intermediate",  5.4,
  "Pulmonary",           "High",          0.8
)

ORD <- dat %>%
  group_by(cause) %>%
  summarise(total = sum(pct), .groups = "drop") %>%
  arrange(total)

dat <- dat %>%
  mutate(
    cause = factor(cause, levels = ORD$cause),
    severity = factor(severity,
                      levels = c("Low", "Intermediate",
                                 "High"))
  )

TOT <- ORD %>%
  mutate(cause = factor(cause, levels = ORD$cause),
         lab = paste0(round(total), "%"))

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = pct, y = cause)) +
  geom_col_pattern(
    aes(fill = severity,
        pattern = severity,
        pattern_angle = severity),
    position = position_stack(reverse = TRUE),
    width = 0.68,
    colour = "white",
    linewidth = 0.4,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_density = 0.12,
    pattern_spacing = 0.020,
    pattern_key_scale_factor = 0.7
  ) +
  geom_text(data = TOT,
            aes(x = total, y = cause, label = lab),
            hjust = -0.25, size = 3.1,
            family = FONT, fontface = "bold",
            colour = DARK, inherit.aes = FALSE) +
  scale_fill_manual(
    values = c("Low" = BLUE,
               "Intermediate" = GREEN,
               "High" = ORANGE),
    name = "PE severity at presentation:"
  ) +
  scale_pattern_manual(
    values = c("Low" = "none",
               "Intermediate" = "stripe",
               "High" = "crosshatch"),
    name = "PE severity at presentation:"
  ) +
  scale_pattern_angle_manual(
    values = c("Low" = 0,
               "Intermediate" = 45,
               "High" = 135),
    name = "PE severity at presentation:"
  ) +
  scale_x_continuous(
    limits = c(0, 38),
    breaks = seq(0, 35, 5),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = paste0("Cause of Inpatient Death After ",
                   "Acute Pulmonary Embolism, by PE ",
                   "Severity"),
    x = "Share of inpatient deaths",
    y = NULL
  ) +
  guides(
    fill = guide_legend(nrow = 1, byrow = TRUE,
                        label.position = "right"),
    pattern = guide_legend(nrow = 1, byrow = TRUE,
                           label.position = "right"),
    pattern_angle = guide_legend(nrow = 1, byrow = TRUE,
                                 label.position = "right")
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 12,
                              hjust = 0.5,
                              margin = margin(b = 12)),
    plot.title.position = "plot",
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 6)),
    axis.text.y = element_text(size = 10, colour = DARK),
    axis.text.x = element_text(size = 9, colour = GREY),
    panel.grid.major.x = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.title = element_text(
      size = 9, colour = DARK,
      margin = margin(r = 12)),
    legend.text = element_text(
      size = 9, colour = DARK,
      margin = margin(l = 4, r = 22)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    legend.box.spacing = unit(0.15, "cm"),
    plot.margin = margin(6, 12, 4, 4)
  )

save_fig(p, "fig1_cause_of_death", width = 6.8,
         height = 3.5)






























# ---------------------------------------------------------
# fig2_cause_by_survival.R
# Causes of death by time from PE diagnosis
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
SKY    <- "#56B4E9"
ORANGE <- "#D55E00"
YELLOW <- "#E69F00"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
GREY   <- "#999999"
DARK   <- "#333333"

LABEL_MIN <- 7

LEG_ROWS <- 3

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded to sum to 100 within each window; the per-window
## percentages are not given in the paper text, so the
## values are approximate.

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         "Over 3 years\n(n = 106)")

CAUSE <- c("PE or associated complications",
           "Recurrent PE",
           "Cancer",
           "Infections",
           "Cardiovascular events",
           "Pulmonary causes",
           "Bleeding (intracranial included)",
           "Suicide",
           "Other",
           "Not determinable")

dat <- tribble(
  ~cause, ~w, ~pct,
  "PE or associated complications",     1, 67,
  "PE or associated complications",     2,  1,
  "PE or associated complications",     3,  0,
  "PE or associated complications",     4,  0,
  "Recurrent PE",                       1,  4,
  "Recurrent PE",                       2,  4,
  "Recurrent PE",                       3,  6,
  "Recurrent PE",                       4,  2,
  "Cancer",                             1, 16,
  "Cancer",                             2, 56,
  "Cancer",                             3, 32,
  "Cancer",                             4, 11,
  "Infections",                         1,  9,
  "Infections",                         2, 13,
  "Infections",                         3, 12,
  "Infections",                         4, 20,
  "Cardiovascular events",              1,  2,
  "Cardiovascular events",              2,  8,
  "Cardiovascular events",              3, 14,
  "Cardiovascular events",              4, 19,
  "Pulmonary causes",                   1,  0,
  "Pulmonary causes",                   2,  3,
  "Pulmonary causes",                   3,  4,
  "Pulmonary causes",                   4,  4,
  "Bleeding (intracranial included)",   1,  1,
  "Bleeding (intracranial included)",   2,  2,
  "Bleeding (intracranial included)",   3,  3,
  "Bleeding (intracranial included)",   4,  4,
  "Suicide",                            1,  0,
  "Suicide",                            2,  2,
  "Suicide",                            3,  0,
  "Suicide",                            4,  0,
  "Other",                              1,  1,
  "Other",                              2,  3,
  "Other",                              3,  3,
  "Other",                              4,  9,
  "Not determinable",                   1,  0,
  "Not determinable",                   2,  8,
  "Not determinable",                   3, 26,
  "Not determinable",                   4, 31
) %>%
  mutate(
    cause = factor(cause, levels = CAUSE),
    win   = factor(WIN[w], levels = WIN)
  )

LABS <- dat %>%
  arrange(win, cause) %>%
  group_by(win) %>%
  mutate(top = cumsum(pct),
         mid = top - pct / 2) %>%
  ungroup() %>%
  filter(pct >= LABEL_MIN) %>%
  mutate(lab = paste0(pct, "%"))

FILLS <- setNames(
  c(BLUE, SKY, ORANGE, YELLOW, GREEN,
    GREEN, PURPLE, PURPLE, SKY, GREY),
  CAUSE)

PATS <- setNames(
  c("none", "stripe", "none", "stripe", "none",
    "crosshatch", "none", "stripe", "circle",
    "crosshatch"),
  CAUSE)

ANGS <- setNames(
  c(0, 45, 0, 135, 0,
    45, 0, 45, 0, 0),
  CAUSE)

SPACE <- setNames(
  c(0.02, 0.030, 0.02, 0.018, 0.02,
    0.024, 0.02, 0.014, 0.030, 0.020),
  CAUSE)

DENS <- setNames(
  c(0.10, 0.16, 0.10, 0.12, 0.10,
    0.10, 0.10, 0.20, 0.30, 0.14),
  CAUSE)

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = win, y = pct)) +
  geom_col_pattern(
    aes(fill = cause,
        pattern = cause,
        pattern_angle = cause,
        pattern_spacing = cause,
        pattern_density = cause),
    position = position_stack(reverse = TRUE),
    width = 0.70,
    colour = "white",
    linewidth = 0.35,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_key_scale_factor = 0.5
  ) +
  geom_label(
    data = LABS,
    aes(x = win, y = mid, label = lab),
    inherit.aes = FALSE,
    fill = "white",
    colour = DARK,
    alpha = 0.88,
    label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 3.0,
    fontface = "bold",
    family = FONT
  ) +
  scale_fill_manual(values = FILLS, name = NULL) +
  scale_pattern_manual(values = PATS, name = NULL) +
  scale_pattern_angle_manual(values = ANGS,
                             name = NULL) +
  scale_pattern_spacing_manual(values = SPACE,
                               name = NULL) +
  scale_pattern_density_manual(values = DENS,
                               name = NULL) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("Cause of Death by Time From ",
                   "Pulmonary Embolism Diagnosis"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths within the window"
  ) +
  guides(
    fill = guide_legend(nrow = LEG_ROWS, byrow = TRUE),
    pattern = guide_legend(nrow = LEG_ROWS,
                           byrow = TRUE),
    pattern_angle = guide_legend(nrow = LEG_ROWS,
                                 byrow = TRUE),
    pattern_spacing = guide_legend(nrow = LEG_ROWS,
                                   byrow = TRUE),
    pattern_density = guide_legend(nrow = LEG_ROWS,
                                   byrow = TRUE)
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 12,
                              hjust = 0.5,
                              margin = margin(b = 12)),
    plot.title.position = "plot",
    axis.title.x = element_text(size = 9.5, colour = DARK,
                                margin = margin(t = 8)),
    axis.title.y = element_text(size = 9.5, colour = DARK,
                                margin = margin(r = 6)),
    axis.text = element_text(size = 9.5, colour = DARK),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.text = element_text(size = 8.4,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 10)),
    legend.key.size = unit(0.36, "cm"),
    legend.margin = margin(8, 0, 0, 0),
    legend.box.spacing = unit(0.25, "cm"),
    plot.margin = margin(6, 6, 4, 4)
  )

save_fig(p, "fig2_cause_by_survival", width = 7.2,
         height = 5.0)



























# ---------------------------------------------------------
# fig2_cause_by_survival.R
# Causes of death by time from PE diagnosis
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
SKY    <- "#56B4E9"
ORANGE <- "#D55E00"
YELLOW <- "#E69F00"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
GREY   <- "#999999"
DARK   <- "#333333"

LABEL_MIN <- 7
LEG_ROWS <- 3

## ---- 2. data --------------------------------------------
## Values are read off Eckelt et al. (2023) Figure 2 and
## rounded to sum to 100 within each window; the per-window
## percentages are not given in the paper text, so the
## values are approximate.

WIN <- c("0-30 days\n(n = 78)",
         "31-365 days\n(n = 97)",
         "1-3 years\n(n = 80)",
         "Over 3 years\n(n = 106)")

CAUSE <- c("PE or associated complications",
           "Recurrent PE",
           "Cancer",
           "Infections",
           "Cardiovascular events",
           "Pulmonary causes",
           "Bleeding (intracranial included)",
           "Suicide",
           "Other",
           "Not determinable")

dat <- tribble(
  ~cause, ~w, ~pct,
  "PE or associated complications",     1, 67,
  "PE or associated complications",     2,  1,
  "PE or associated complications",     3,  0,
  "PE or associated complications",     4,  0,
  "Recurrent PE",                       1,  4,
  "Recurrent PE",                       2,  4,
  "Recurrent PE",                       3,  6,
  "Recurrent PE",                       4,  2,
  "Cancer",                             1, 16,
  "Cancer",                             2, 56,
  "Cancer",                             3, 32,
  "Cancer",                             4, 11,
  "Infections",                         1,  9,
  "Infections",                         2, 13,
  "Infections",                         3, 12,
  "Infections",                         4, 20,
  "Cardiovascular events",              1,  2,
  "Cardiovascular events",              2,  8,
  "Cardiovascular events",              3, 14,
  "Cardiovascular events",              4, 19,
  "Pulmonary causes",                   1,  0,
  "Pulmonary causes",                   2,  3,
  "Pulmonary causes",                   3,  4,
  "Pulmonary causes",                   4,  4,
  "Bleeding (intracranial included)",   1,  1,
  "Bleeding (intracranial included)",   2,  2,
  "Bleeding (intracranial included)",   3,  3,
  "Bleeding (intracranial included)",   4,  4,
  "Suicide",                            1,  0,
  "Suicide",                            2,  2,
  "Suicide",                            3,  0,
  "Suicide",                            4,  0,
  "Other",                              1,  1,
  "Other",                              2,  3,
  "Other",                              3,  3,
  "Other",                              4,  9,
  "Not determinable",                   1,  0,
  "Not determinable",                   2,  8,
  "Not determinable",                   3, 26,
  "Not determinable",                   4, 31
) %>%
  mutate(
    cause = factor(cause, levels = CAUSE),
    win   = factor(WIN[w], levels = WIN)
  )

LABS <- dat %>%
  arrange(win, cause) %>%
  group_by(win) %>%
  mutate(top = cumsum(pct),
         mid = top - pct / 2) %>%
  ungroup() %>%
  filter(pct >= LABEL_MIN) %>%
  mutate(lab = paste0(pct, "%"))

FILLS <- setNames(
  c(BLUE, SKY, ORANGE, YELLOW, GREEN,
    GREEN, PURPLE, PURPLE, SKY, GREY),
  CAUSE)

PATS <- setNames(
  c("none", "stripe", "none", "stripe", "none",
    "crosshatch", "none", "stripe", "circle",
    "crosshatch"),
  CAUSE)

ANGS <- setNames(
  c(0, 45, 0, 135, 0,
    45, 0, 45, 0, 0),
  CAUSE)

SPACE <- setNames(
  c(0.02, 0.030, 0.02, 0.018, 0.02,
    0.024, 0.02, 0.014, 0.030, 0.020),
  CAUSE)

DENS <- setNames(
  c(0.10, 0.16, 0.10, 0.12, 0.10,
    0.10, 0.10, 0.20, 0.30, 0.14),
  CAUSE)

## ---- 3. plot --------------------------------------------

p <- ggplot(dat, aes(x = win, y = pct)) +
  geom_col_pattern(
    aes(fill = cause,
        pattern = cause,
        pattern_angle = cause,
        pattern_spacing = cause,
        pattern_density = cause),
    position = position_stack(reverse = TRUE),
    width = 0.70,
    colour = "white",
    linewidth = 0.35,
    pattern_fill = "white",
    pattern_colour = "white",
    pattern_key_scale_factor = 0.5
  ) +
  geom_label(
    data = LABS,
    aes(x = win, y = mid, label = lab),
    inherit.aes = FALSE,
    fill = "white",
    colour = DARK,
    alpha = 0.88,
    label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 3.0,
    fontface = "bold",
    family = FONT
  ) +
  scale_fill_manual(values = FILLS, name = NULL) +
  scale_pattern_manual(values = PATS, name = NULL) +
  scale_pattern_angle_manual(values = ANGS,
                             name = NULL) +
  scale_pattern_spacing_manual(values = SPACE,
                               name = NULL) +
  scale_pattern_density_manual(values = DENS,
                               name = NULL) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title = paste0("Cause of Death by Time From ",
                   "Pulmonary Embolism Diagnosis"),
    x = "Time from PE diagnosis to death",
    y = "Share of deaths within the window"
  ) +
  guides(
    fill = guide_legend(nrow = LEG_ROWS, byrow = TRUE),
    pattern = guide_legend(nrow = LEG_ROWS,
                           byrow = TRUE),
    pattern_angle = guide_legend(nrow = LEG_ROWS,
                                 byrow = TRUE),
    pattern_spacing = guide_legend(nrow = LEG_ROWS,
                                   byrow = TRUE),
    pattern_density = guide_legend(nrow = LEG_ROWS,
                                   byrow = TRUE)
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 12,
                              hjust = 0.5,
                              margin = margin(b = 12)),
    plot.title.position = "plot",
    axis.title.x = element_text(size = 10,
                                colour = DARK,
                                margin = margin(t = 2)),
    axis.title.y = element_text(size = 10,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text.x = element_text(size = 9.5,
                               colour = DARK,
                               margin = margin(t = 2)),
    axis.text.y = element_text(size = 9.5,
                               colour = DARK,
                               margin = margin(r = 2)),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.text = element_text(size = 8.4,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 10)),
    legend.key.size = unit(0.36, "cm"),
    legend.margin = margin(4, 0, 0, 0),
    legend.box.spacing = unit(0.12, "cm"),
    plot.margin = margin(6, 6, 4, 2)
  )

save_fig(p, "fig2_cause_by_survival", width = 7.2,
         height = 4.8)

































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

BLUE <- "#0072B2"
DARK <- "#333333"
GREY <- "#7F7F7F"

## Fill lightness ordered Q1 (pale) to Q4 (full).
QFILL <- c(Q1 = "#C6DBEF",
           Q2 = "#6BAED6",
           Q3 = "#2B7BBA",
           Q4 = BLUE)

QSPACE <- c(Q1 = NA,
            Q2 = 0.030,
            Q3 = 0.018,
            Q4 = 0.014)

## ---- 2. data --------------------------------------------

OUTCOME <- "30-day death"

WANT <- c("Urea nitrogen", "rr", "Age", "hr",
          "aniongap", "White cell count")

## the export left some names untranslated
RENAME <- c("rr" = "Respiratory rate",
            "hr" = "Heart rate",
            "aniongap" = "Anion gap",
            "sbp" = "Systolic BP",
            "dbp" = "Diastolic BP",
            "plt" = "Platelets",
            "rbc" = "Red cell count",
            "temp" = "Temperature",
            "bicarb" = "Bicarbonate")

d <- read.csv(file.path(data_dir,
                        "quartile_event_rates.csv"),
              stringsAsFactors = FALSE)

d$feature <- ifelse(d$feature %in% names(RENAME),
                    RENAME[d$feature], d$feature)

WANT_DISP <- ifelse(WANT %in% names(RENAME),
                    RENAME[WANT], WANT)
KEEP <- intersect(WANT_DISP, unique(d$feature))

miss <- setdiff(WANT_DISP, KEEP)
if (length(miss) > 0) {
  message("not in CSV: ",
          paste(miss, collapse = ", "))
}

d <- d %>%
  filter(feature %in% KEEP) %>%
  mutate(
    feature = factor(feature, levels = KEEP),
    quartile = factor(quartile,
                      levels = c("Q1", "Q2",
                                 "Q3", "Q4")),
    lab = sprintf("%.3f", rate))

## cohort event rate, drawn as a reference line
BASE <- mean(d$base_rate, na.rm = TRUE)
if (is.na(BASE)) BASE <- 0.117

## ---- 3. hatch rules -------------------------------------

rules <- d %>%
  rowwise() %>%
  do({
    r <- .
    sp <- QSPACE[[as.character(r$quartile)]]
    if (is.na(sp)) {
      data.frame()
    } else {
      ys <- seq(sp, r$rate - sp / 2, by = sp)
      if (length(ys) == 0) {
        data.frame()
      } else {
        data.frame(feature = r$feature,
                   quartile = r$quartile,
                   y = ys)
      }
    }
  }) %>%
  ungroup() %>%
  mutate(feature = factor(feature, levels = KEEP),
         quartile = factor(quartile,
                           levels = c("Q1", "Q2",
                                      "Q3", "Q4")))

## ---- 4. plot --------------------------------------------

p <- ggplot(d, aes(x = quartile, y = rate)) +
  geom_col(aes(fill = quartile), width = 0.72,
           colour = DARK, linewidth = 0.25) +
  geom_segment(
    data = rules,
    aes(x = as.numeric(quartile) - 0.36,
        xend = as.numeric(quartile) + 0.36,
        y = y, yend = y),
    colour = "white", linewidth = 0.4,
    inherit.aes = FALSE) +
  geom_hline(yintercept = BASE, linetype = "22",
             colour = GREY, linewidth = 0.4) +
  geom_text(aes(label = lab), vjust = -0.6,
            size = 2.8, colour = DARK,
            family = FONT) +
  facet_wrap(~ feature, nrow = 2) +
  scale_fill_manual(values = QFILL,
                    guide = "none") +
  scale_y_continuous(
    limits = c(0, 0.30),
    breaks = seq(0, 0.30, 0.05),
    expand = expansion(mult = c(0, 0.02))) +
  labs(
    title = paste0("Observed Event Rate by Quartile of ",
                   "Each Leading Feature"),
    x = "Quartile of feature value",
    y = paste0("Observed ", OUTCOME, " rate")) +
  theme(
    plot.title = element_text(face = "bold", size = 12,
                              hjust = 0.5,
                              margin = margin(b = 12)),
    plot.title.position = "plot",
    strip.text = element_text(face = "bold", size = 10.5,
                              colour = DARK,
                              margin = margin(b = 4)),
    axis.title.x = element_text(size = 12, colour = DARK,
                                margin = margin(t = 2)),
    axis.title.y = element_text(size = 12, colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 9.5, colour = DARK,
                             margin = margin(2, 2, 2, 2)),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(0.7, "lines"),
    plot.margin = margin(6, 6, 4, 2))

save_fig(p, "fig13_quartile_rates",
         width = 7.6, height = 5.4)














d0 <- read.csv(file.path(data_dir,
                         "quartile_event_rates.csv"),
               stringsAsFactors = FALSE)
str(d0)
head(d0, 8)
unique(d0$quartile)





















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

BLUE <- "#0072B2"
DARK <- "#333333"
GREY <- "#7F7F7F"

QLEV <- c("Q1", "Q2", "Q3", "Q4")

QFILL <- c(Q1 = "#C6DBEF",
           Q2 = "#6BAED6",
           Q3 = "#2B7BBA",
           Q4 = BLUE)

QSPACE <- c(Q1 = NA_real_,
            Q2 = 0.030,
            Q3 = 0.018,
            Q4 = 0.014)

## ---- 2. data --------------------------------------------

OUTCOME <- "Composite (30-day)"

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

d0 <- read.csv(file.path(data_dir,
                         "quartile_event_rates.csv"),
               stringsAsFactors = FALSE)

message("outcomes in file: ",
        paste(unique(d0$outcome), collapse = " | "))

if (!OUTCOME %in% d0$outcome) {
  stop("OUTCOME not found. Available: ",
       paste(unique(d0$outcome), collapse = " | "))
}

d <- d0 %>% filter(outcome == OUTCOME)

d$feature <- ifelse(d$feature %in% names(RENAME),
                    RENAME[d$feature], d$feature)

WANT_DISP <- ifelse(WANT %in% names(RENAME),
                    RENAME[WANT], WANT)
KEEP <- intersect(WANT_DISP, unique(d$feature))

miss <- setdiff(WANT_DISP, KEEP)
if (length(miss) > 0) {
  message("not in CSV: ",
          paste(miss, collapse = ", "))
}

d <- d %>%
  filter(feature %in% KEEP) %>%
  mutate(
    feature = factor(feature, levels = KEEP),
    quartile = factor(paste0("Q", quartile),
                      levels = QLEV),
    rate = as.numeric(rate),
    lab = sprintf("%.3f", rate))

BASE <- d0 %>%
  filter(outcome == OUTCOME) %>%
  summarise(b = sum(rate * n) / sum(n)) %>%
  pull(b)

message("cohort event rate: ", round(BASE, 4))

## Headroom above the tallest bar for its value label.
YMAX <- ceiling((max(d$rate) + 0.03) * 20) / 20

## ---- 3. hatch rules -------------------------------------

mk_rules <- function(q, rate) {
  sp <- QSPACE[[q]]
  if (is.na(sp)) return(numeric(0))
  ys <- seq(sp, rate - sp / 2, by = sp)
  if (length(ys) == 0) return(numeric(0))
  ys
}

rules <- do.call(rbind, lapply(seq_len(nrow(d)),
  function(i) {
    ys <- mk_rules(as.character(d$quartile[i]),
                   d$rate[i])
    if (length(ys) == 0) return(NULL)
    data.frame(feature = as.character(d$feature[i]),
               quartile = as.character(d$quartile[i]),
               y = ys,
               stringsAsFactors = FALSE)
  }))

if (!is.null(rules)) {
  rules$feature <- factor(rules$feature,
                          levels = KEEP)
  rules$quartile <- factor(rules$quartile,
                           levels = QLEV)
}

## ---- 4. plot --------------------------------------------

p <- ggplot(d, aes(x = quartile, y = rate)) +
  geom_col(aes(fill = quartile), width = 0.72,
           colour = DARK, linewidth = 0.25)

if (!is.null(rules)) {
  p <- p + geom_segment(
    data = rules,
    aes(x = as.numeric(quartile) - 0.36,
        xend = as.numeric(quartile) + 0.36,
        y = y, yend = y),
    colour = "white", linewidth = 0.4,
    inherit.aes = FALSE)
}

p <- p +
  geom_hline(yintercept = BASE, linetype = "22",
             colour = GREY, linewidth = 0.4) +
  geom_text(aes(label = lab), vjust = -0.6,
            size = 2.8, colour = DARK,
            family = FONT) +
  facet_wrap(~ feature, nrow = 2) +
  scale_fill_manual(values = QFILL,
                    guide = "none") +
  scale_y_continuous(
    limits = c(0, YMAX),
    breaks = seq(0, YMAX, 0.05),
    expand = expansion(mult = c(0, 0.02))) +
  labs(
    title = paste0("Observed Event Rate by Quartile of ",
                   "Each Leading Feature"),
    x = "Quartile of feature value",
    y = "Observed event rate") +
  theme(
    plot.title = element_text(face = "bold", size = 12,
                              hjust = 0.5,
                              margin = margin(b = 12)),
    plot.title.position = "plot",
    strip.text = element_text(face = "bold", size = 10.5,
                              colour = DARK,
                              margin = margin(b = 4)),
    axis.title.x = element_text(size = 12, colour = DARK,
                                margin = margin(t = 2)),
    axis.title.y = element_text(size = 12, colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 9.5, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(0.7, "lines"),
    plot.margin = margin(6, 6, 4, 2))

save_fig(p, "fig13_quartile_rates",
         width = 7.6, height = 5.4)



























# ---------------------------------------------------------
# fig14_rank_concordance.R
# Feature attribution rank at the source institution
# against rank at the target institution
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

BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"

## Features shifting this many ranks or more get labelled.
SHIFT_LAB <- 3

## ---- 2. data --------------------------------------------

NICE <- c(rr = "Respiratory rate",
          hr = "Heart rate",
          aniongap = "Anion gap",
          sbp = "Systolic BP",
          dbp = "Diastolic BP",
          plt = "Platelets",
          rbc = "Red cell count",
          temp = "Temperature",
          bicarb = "Bicarbonate",
          bun = "Urea nitrogen",
          hct = "Haematocrit",
          hgb = "Haemoglobin",
          copd = "COPD",
          cancer = "Malignancy",
          heart_failure = "Heart failure",
          afib = "Atrial fibrillation")

d <- read.csv(file.path(data_dir,
                        "shap_rank_concordance.csv"),
              stringsAsFactors = FALSE)

message("columns: ", paste(names(d), collapse = ", "))

d <- d %>%
  mutate(
    label = ifelse(feature %in% names(NICE),
                   NICE[feature], feature),
    shift = abs(rank_source - rank_target),
    big = shift >= SHIFT_LAB)

rho <- d %>%
  group_by(outcome) %>%
  summarise(
    r = cor(rank_source, rank_target,
            method = "spearman"),
    .groups = "drop") %>%
  mutate(lab = sprintf("Spearman rho = %.3f", r))

message("rho: ",
        paste(sprintf("%s %.3f", rho$outcome, rho$r),
              collapse = " | "))

## ---- 3. plot --------------------------------------------

f <- ggplot(d, aes(x = rank_source, y = rank_target)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "22", colour = "grey55",
              linewidth = 0.4) +
  geom_point(aes(colour = big, shape = big,
                 fill = big),
             size = 2.1, stroke = 0.7) +
  geom_text(data = filter(d, big),
            aes(label = label),
            hjust = -0.15, size = 2.8,
            colour = ORANGE, family = FONT) +
  geom_text(data = rho,
            aes(x = 2, y = 27, label = lab),
            hjust = 0, size = 3.1, colour = "grey25",
            family = FONT, inherit.aes = FALSE) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(
    values = c("FALSE" = BLUE, "TRUE" = ORANGE),
    guide = "none") +
  scale_fill_manual(
    values = c("FALSE" = BLUE, "TRUE" = NA),
    guide = "none") +
  scale_shape_manual(
    values = c("FALSE" = 21, "TRUE" = 22),
    guide = "none") +
  scale_x_continuous(
    limits = c(0, 30), breaks = seq(0, 28, by = 7),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(0, 30), breaks = seq(0, 28, by = 7),
    expand = c(0, 0)) +
  coord_fixed() +
  labs(x = "Rank at the source institution",
       y = "Rank at the target institution",
       title = paste("Feature Attribution Rank at the",
                     "Source Against the Target",
                     "Institution")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 10.5,
                                  face = "bold",
                                  colour = DARK,
                                  margin = margin(b = 4)),
        axis.title.x = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(t = 2)),
        axis.title.y = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(r = 2)),
        axis.text = element_text(size = 9.5,
                                 colour = DARK),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 12)),
        plot.title.position = "plot",
        panel.spacing = unit(0.8, "lines"),
        plot.margin = margin(6, 8, 4, 4))

cat("\nwriting figure...\n")
print(system.time(
  save_fig(f, "fig14_rank_concordance", 8.0, 4.4)))
































# ---------------------------------------------------------
# fig10_ecg_statements.R
# Statement-level coefficients in the ECG modality
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

BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

N_SHOW <- 12

## ---- 2. SCP-ECG statement names -------------------------

SCP <- c(
  AFIB  = "Atrial fibrillation",
  AFLT  = "Atrial flutter",
  ALMI  = "Anterolateral MI",
  AMI   = "Anterior MI",
  ASMI  = "Anteroseptal MI",
  CLBBB = "Complete LBBB",
  CRBBB = "Complete RBBB",
  HVOLT = "High QRS voltage",
  ILBBB = "Incomplete LBBB",
  ILMI  = "Inferolateral MI",
  IMI   = "Inferior MI",
  INJAL = "Anterolateral injury",
  INJIL = "Inferolateral injury",
  IPLMI = "Inferoposterolateral MI",
  IRBBB = "Incomplete RBBB",
  LAFB  = "Left anterior fascicular block",
  LMI   = "Lateral MI",
  LVH   = "Left ventricular hypertrophy",
  LVOLT = "Low QRS voltage",
  NST_  = "ST changes, non-specific",
  PAC   = "Atrial premature complex",
  PACE  = "Artificial pacemaker",
  PVC   = "Ventricular premature complex",
  RAO.RAE = "Right atrial enlargement",
  `RAO/RAE` = "Right atrial enlargement",
  SARRH = "Sinus arrhythmia",
  SBRAD = "Sinus bradycardia",
  SR    = "Sinus rhythm",
  STACH = "Sinus tachycardia",
  TAB_  = "T-wave abnormality",
  VCLVH = "Voltage criteria for LVH"
)

## ---- 3. panel builder -----------------------------------

ecg_panel <- function(fn, title) {
  d <- read.csv(file.path(data_dir, fn),
                stringsAsFactors = FALSE)

  message(fn, " columns: ",
          paste(names(d), collapse = ", "))

  ccol <- intersect(c("code", "statement", "feature"),
                    names(d))[1]
  bcol <- intersect(c("beta", "coef", "coefficient"),
                    names(d))[1]
  scol <- intersect(c("mean_abs_shap", "abs_shap",
                      "importance"),
                    names(d))[1]
  if (is.na(ccol) || is.na(bcol)) {
    stop("need a code column and a beta column; found: ",
         paste(names(d), collapse = ", "))
  }

  d$code <- as.character(d[[ccol]])
  d$beta <- as.numeric(d[[bcol]])
  d$ord <- if (is.na(scol)) abs(d$beta) else
    as.numeric(d[[scol]])

  miss <- setdiff(d$code[seq_len(min(N_SHOW,
                                     nrow(d)))],
                  names(SCP))
  if (length(miss) > 0) {
    message("no SCP name for: ",
            paste(miss, collapse = ", "))
  }

  d <- d %>%
    arrange(desc(ord)) %>%
    slice_head(n = N_SHOW) %>%
    mutate(
      name = ifelse(code %in% names(SCP),
                    SCP[code], code),
      label = paste0(name, "  (", code, ")"),
      label = factor(label,
                     levels = rev(label)),
      dir = ifelse(beta >= 0, "Raises risk",
                   "Lowers risk"))

  LIM <- max(abs(d$beta)) * 1.35

  ggplot(d, aes(x = beta, y = label)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = beta,
                     y = label, yend = label,
                     colour = dir),
                 linewidth = 0.7) +
    geom_point(aes(colour = dir, shape = dir),
               size = 2.6, fill = "white",
               stroke = 0.8) +
    geom_text(aes(label = sprintf("%+.3f", beta),
                  hjust = ifelse(beta >= 0,
                                 -0.28, 1.28)),
              size = 2.7, colour = DARK,
              family = FONT) +
    scale_colour_manual(
      values = c("Raises risk" = ORANGE,
                 "Lowers risk" = BLUE),
      guide = "none") +
    scale_shape_manual(
      values = c("Raises risk" = 24,
                 "Lowers risk" = 25),
      guide = "none") +
    scale_x_continuous(
      limits = c(-LIM, LIM),
      expand = expansion(mult = c(0, 0))) +
    labs(
      x = "Coefficient (standardised logit units)",
      y = NULL,
      title = title) +
    theme(
      plot.title = element_text(face = "bold",
                                size = 12,
                                hjust = 0.5,
                                margin = margin(b = 12)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 2)),
      axis.text.y = element_text(size = 10.5,
                                 colour = DARK),
      axis.text.x = element_text(size = 9.5,
                                 colour = GREY),
      axis.ticks.length = unit(0.06, "cm"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      plot.margin = margin(6, 8, 4, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 6.6
HT <- 4.6

a <- ecg_panel(
  "table12_ecg_composite_30d.csv",
  paste0("Statement-Level Coefficients in the ECG ",
         "Modality: Composite Outcome"))
save_fig(a, "fig10a_ecg_composite",
         width = WD, height = HT)

b <- ecg_panel(
  "table12_ecg_death_30d.csv",
  paste0("Statement-Level Coefficients in the ECG ",
         "Modality: 30-Day Death"))
save_fig(b, "fig10b_ecg_death_30d",
         width = WD, height = HT)

c <- ecg_panel(
  "table12_ecg_cv_first.csv",
  paste0("Statement-Level Coefficients in the ECG ",
         "Modality: Cardiovascular Readmission"))
save_fig(c, "fig10c_ecg_cv_first",
         width = WD, height = HT)






















# ---------------------------------------------------------
# fig10_ecg_statements.R
# Named SCP-ECG statement contributions
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

BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

N_TOP <- 12

## ---- 2. SCP-ECG statement decode ------------------------

scp <- c(
  NORM = "Normal ECG",
  SR = "Sinus rhythm",
  AFIB = "Atrial fibrillation",
  AFLT = "Atrial flutter",
  STACH = "Sinus tachycardia",
  SBRAD = "Sinus bradycardia",
  SARRH = "Sinus arrhythmia",
  SVARR = "Supraventricular arrhythmia",
  SVTAC = "Supraventricular tachycardia",
  PSVT = "Paroxysmal SVT",
  PACE = "Artificial pacemaker",
  BIGU = "Bigeminal pattern",
  TRIGU = "Trigeminal pattern",
  PAC = "Atrial premature complex",
  PVC = "Ventricular premature complex",
  "PRC(S)" = "Premature complexes",
  IMI = "Inferior MI",
  ASMI = "Anteroseptal MI",
  AMI = "Anterior MI",
  ALMI = "Anterolateral MI",
  LMI = "Lateral MI",
  ILMI = "Inferolateral MI",
  IPMI = "Inferoposterior MI",
  IPLMI = "Inferoposterolateral MI",
  PMI = "Posterior MI",
  ANEUR = "ST-T changes, aneurysm",
  QWAVE = "Q waves present",
  ABQRS = "Abnormal QRS",
  "1AVB" = "First-degree AV block",
  "2AVB" = "Second-degree AV block",
  "3AVB" = "Complete AV block",
  LPR = "Prolonged PR interval",
  IRBBB = "Incomplete RBBB",
  CRBBB = "Complete RBBB",
  ILBBB = "Incomplete LBBB",
  CLBBB = "Complete LBBB",
  LAFB = "Left anterior fascicular block",
  LPFB = "Left posterior fascicular block",
  IVCD = "Intraventricular conduction delay",
  WPW = "Wolff-Parkinson-White",
  LVH = "Left ventricular hypertrophy",
  RVH = "Right ventricular hypertrophy",
  VCLVH = "Voltage criteria for LVH",
  SEHYP = "Septal hypertrophy",
  "LAO/LAE" = "Left atrial enlargement",
  "RAO/RAE" = "Right atrial enlargement",
  LVOLT = "Low QRS voltage",
  HVOLT = "High QRS voltage",
  "ISC_" = "Ischaemia, non-specific",
  ISCAL = "Ischaemia, anterolateral",
  ISCAS = "Ischaemia, anteroseptal",
  ISCAN = "Ischaemia, anterior",
  ISCIN = "Ischaemia, inferior",
  ISCIL = "Ischaemia, inferolateral",
  ISCLA = "Ischaemia, lateral",
  INJAS = "Injury, anteroseptal",
  INJAL = "Injury, anterolateral",
  INJIN = "Injury, inferior",
  INJIL = "Injury, inferolateral",
  INJLA = "Injury, lateral",
  "STD_" = "ST depression, non-specific",
  "STE_" = "ST elevation, non-specific",
  "NST_" = "ST changes, non-specific",
  "NT_" = "T-wave changes, non-specific",
  NDT = "Non-diagnostic T abnormality",
  "TAB_" = "T-wave abnormality",
  INVT = "Inverted T-waves",
  LOWT = "Low-amplitude T-waves",
  LNGQT = "Long QT interval",
  DIG = "Digitalis effect",
  EL = "Electrolyte or drug effect")

lab_of <- function(code) {
  out <- unname(scp[code])
  ifelse(is.na(out), code, out)
}

## ---- 3. one panel ---------------------------------------

panel_scp <- function(outcome, title, n_top = N_TOP) {

  d <- read.csv(file.path(
    data_dir, paste0("table12_ecg_", outcome, ".csv")),
    stringsAsFactors = FALSE)

  message(outcome, " columns: ",
          paste(names(d), collapse = ", "))

  d <- d %>%
    arrange(desc(mean_abs_shap)) %>%
    slice_head(n = n_top) %>%
    mutate(label = paste0(lab_of(code), "  (", code, ")"),
           label = factor(label,
                          levels = rev(label)),
           direction = ifelse(beta > 0, "Raises risk",
                              "Lowers risk"))

  miss <- setdiff(d$code, names(scp))
  if (length(miss) > 0) {
    message("  no decode for: ",
            paste(miss, collapse = ", "))
  }

  LIM <- max(abs(d$beta)) * 1.38

  ggplot(d, aes(x = beta, y = label)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = beta, yend = label,
                     colour = direction),
                 linewidth = 0.9, alpha = 0.8) +
    geom_point(aes(colour = direction,
                   shape = direction),
               size = 2.6, fill = "white",
               stroke = 0.8) +
    geom_text(aes(label = sprintf("%+.3f", beta),
                  hjust = ifelse(beta >= 0,
                                 -0.30, 1.30)),
              size = 2.7, colour = DARK,
              family = FONT) +
    scale_colour_manual(
      values = c("Raises risk" = ORANGE,
                 "Lowers risk" = BLUE),
      guide = "none") +
    scale_shape_manual(
      values = c("Raises risk" = 24,
                 "Lowers risk" = 25),
      guide = "none") +
    scale_x_continuous(
      limits = c(-LIM, LIM),
      expand = expansion(mult = c(0, 0))) +
    labs(x = "Coefficient (standardised logit units)",
         y = NULL, title = title) +
    theme(
      plot.title = element_text(size = 12,
                                face = "bold",
                                hjust = 0.5,
                                margin = margin(b = 12)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 2)),
      axis.text.y = element_text(size = 10,
                                 colour = DARK),
      axis.text.x = element_text(size = 9.5,
                                 colour = GREY),
      axis.ticks.length = unit(0.06, "cm"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      plot.margin = margin(6, 10, 4, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 6.8
HT <- 4.6

p1 <- panel_scp(
  "composite_30d",
  paste0("Statement-Level Coefficients in the ECG ",
         "Modality: Composite Outcome"))
save_fig(p1, "fig10a_ecg_composite",
         width = WD, height = HT)

p2 <- panel_scp(
  "death_30d",
  paste0("Statement-Level Coefficients in the ECG ",
         "Modality: 30-Day Death"))
save_fig(p2, "fig10b_ecg_death_30d",
         width = WD, height = HT)

p3 <- panel_scp(
  "cv_first",
  paste0("Statement-Level Coefficients in the ECG ",
         "Modality: Cardiovascular Readmission"))
save_fig(p3, "fig10c_ecg_cv_first",
         width = WD, height = HT)





























# ---------------------------------------------------------
# fig11_ctpa_blocks.R
# Feature contributions in the CTPA report modality,
# grouped by semantic block
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

DARK <- "#333333"
GREY <- "#7F7F7F"

N_TOP <- 15

## ---- 2. feature decode ----------------------------------

nice <- c(
  pe_pos = "Pulmonary embolus present",
  pe_neg = "No pulmonary embolus",
  saddle = "Saddle embolus",
  central = "Central embolus",
  lobar = "Lobar embolus",
  segmental = "Segmental embolus",
  subseg = "Subsegmental embolus",
  bilateral = "Bilateral emboli",
  rv_strain = "Right ventricular strain",
  septal_bow = "Septal bowing",
  reflux = "Contrast reflux into IVC",
  mpa_enlarge = "Enlarged pulmonary artery",
  infarct = "Pulmonary infarct",
  effusion = "Pleural effusion",
  malignancy = "Malignancy",
  consolid = "Consolidation",
  atelect = "Atelectasis",
  edema = "Pulmonary oedema",
  cardiomeg = "Cardiomegaly",
  adenopathy = "Lymphadenopathy",
  cm_obesity = "Obesity",
  cm_aortic_ath = "Aortic atherosclerosis",
  cm_emphysema = "Emphysema",
  cm_mets = "Metastatic disease",
  cm_pericard_eff = "Pericardial effusion",
  cm_fibrosis = "Pulmonary fibrosis",
  cm_ascites = "Ascites",
  cm_vert_fx = "Vertebral fracture",
  cm_aneurysm = "Aortic aneurysm",
  cm_cirrhosis = "Cirrhosis",
  cm_valve = "Valvular disease",
  cm_bronchiect = "Bronchiectasis",
  cm_pleural_thick = "Pleural thickening",
  cm_lymphangitic = "Lymphangitic spread",
  cm_renal = "Renal disease",
  cm_cachexia = "Cachexia",
  dv_ett = "Endotracheal tube",
  dv_trach = "Tracheostomy",
  dv_cvc = "Central venous catheter",
  dv_ngtube = "Nasogastric tube",
  dv_pacer = "Pacemaker or ICD",
  dv_sternotomy = "Sternotomy wires",
  dv_ivcfilter = "IVC filter",
  dv_chesttube = "Chest tube",
  txt_len = "Report length",
  n_sent = "Sentence count")

lab_of <- function(f) {
  out <- unname(nice[f])
  ifelse(is.na(out), f, out)
}

BLOCKS <- c("PE descriptor", "Incidental finding",
            "Comorbidity", "Support device",
            "Report metadata")

## Okabe-Ito
PAL <- c("PE descriptor"      = "#0072B2",
         "Incidental finding" = "#D55E00",
         "Comorbidity"        = "#009E73",
         "Support device"     = "#CC79A7",
         "Report metadata"    = "#999999")

SHP <- c("PE descriptor"      = 16,
         "Incidental finding" = 17,
         "Comorbidity"        = 15,
         "Support device"     = 18,
         "Report metadata"    = 4)

## ---- 3. one panel ---------------------------------------

panel_ctpa <- function(outcome, title, n_top = N_TOP) {

  d <- read.csv(file.path(
    data_dir, paste0("ctpa_blocks_", outcome, ".csv")),
    stringsAsFactors = FALSE)

  tot <- sum(d$mean_abs_shap)
  bl <- d %>%
    group_by(block) %>%
    summarise(n = n(),
              share = 100 * sum(mean_abs_shap) / tot,
              per_feat = sum(mean_abs_shap) / n(),
              mean_beta = mean(beta),
              .groups = "drop") %>%
    arrange(desc(share))

  cat("\n---", outcome, "block summary ---\n")
  print(as.data.frame(bl), digits = 3)
  cat("max |beta|:",
      sprintf("%.4f", max(abs(d$beta))), "\n")

  d <- d %>%
    arrange(desc(mean_abs_shap)) %>%
    slice_head(n = n_top) %>%
    mutate(label = lab_of(feature),
           label = factor(label, levels = rev(label)),
           block = factor(block, levels = BLOCKS))

  LIM <- max(abs(d$beta)) * 1.42

  ggplot(d, aes(x = beta, y = label, colour = block)) +
    geom_vline(xintercept = 0, colour = DARK,
               linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = beta, yend = label),
                 linewidth = 0.9, alpha = 0.8) +
    geom_point(aes(shape = block), size = 2.8) +
    geom_text(aes(label = sprintf("%+.3f", beta),
                  hjust = ifelse(beta >= 0,
                                 -0.28, 1.28)),
              size = 2.6, colour = DARK,
              family = FONT, show.legend = FALSE) +
    scale_colour_manual(values = PAL, drop = FALSE,
                        name = NULL) +
    scale_shape_manual(values = SHP, drop = FALSE,
                       name = NULL) +
    scale_x_continuous(
      limits = c(-LIM, LIM),
      expand = expansion(mult = c(0, 0))) +
    labs(x = "Coefficient (standardised units)",
         y = NULL, title = title) +
    guides(colour = guide_legend(nrow = 2),
           shape = guide_legend(nrow = 2)) +
    theme(
      plot.title = element_text(size = 12,
                                face = "bold",
                                hjust = 0.5,
                                margin = margin(b = 12)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 2)),
      axis.text.y = element_text(size = 10,
                                 colour = DARK),
      axis.text.x = element_text(size = 9.5,
                                 colour = GREY),
      axis.ticks.length = unit(0.06, "cm"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      legend.position = "bottom",
      legend.justification = "center",
      legend.text = element_text(size = 9,
                                 colour = DARK,
                                 margin = margin(l = 3,
                                                 r = 12)),
      legend.key.size = unit(0.36, "cm"),
      legend.margin = margin(4, 0, 0, 0),
      legend.box.spacing = unit(0.12, "cm"),
      plot.margin = margin(6, 10, 4, 4))
}

## ---- 4. build and save ----------------------------------

WD <- 7.0
HT <- 5.4

c1 <- panel_ctpa(
  "death_30d",
  paste0("Feature Contributions in the CTPA Report ",
         "Modality: 30-Day Death"))
save_fig(c1, "fig11a_ctpa_death_30d",
         width = WD, height = HT)

c2 <- panel_ctpa(
  "composite_30d",
  paste0("Feature Contributions in the CTPA Report ",
         "Modality: Composite Outcome"))
save_fig(c2, "fig11b_ctpa_composite",
         width = WD, height = HT)

c3 <- panel_ctpa(
  "cv_first",
  paste0("Feature Contributions in the CTPA Report ",
         "Modality: Cardiovascular Readmission"))
save_fig(c3, "fig11c_ctpa_cv_first",
         width = WD, height = HT)




























# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## ---- 2. shared theme ------------------------------------

house <- function() {
  theme(
    plot.title = element_text(face = "bold",
                              size = 12,
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 10, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )
}

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

ACOUNT <- tribble(
  ~endpoint,        ~n,   ~ev,
  "Composite",      1027, 210,
  "30-day death",   1027, 161,
  "CV readmission",  871,  54
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         lab = paste0("n = ", format(n, big.mark = ","),
                      "; ", ev, " events"))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    data = ACOUNT,
    aes(x = endpoint, y = 0.762, label = lab),
    inherit.aes = FALSE,
    size = 2.6, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.78)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Against Acquisition Context"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 4.0)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## Source: data/cxr_strata_ci.csv
##
## Cardiovascular readmission is excluded: 19, 15 and 20
## events per stratum give intervals too wide to interpret.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,   ~lo,    ~hi,    ~ev,
  "Composite",    "All portable",  0.6303, 0.5747, 0.6845, 119,
  "Composite",    "Mixed",         0.6462, 0.5585, 0.7333,  47,
  "Composite",    "None portable", 0.6817, 0.5995, 0.7680,  44,
  "30-day death", "All portable",  0.6456, 0.5865, 0.7007, 103,
  "30-day death", "Mixed",         0.6441, 0.5415, 0.7470,  32,
  "30-day death", "None portable", 0.6788, 0.5586, 0.7888,  26
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0(ev, " ev"))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_B,
    width = 0.14, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(y = hi, label = nlab),
    position = DODGE_B,
    vjust = -0.7,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.85, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.85)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Within Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 5.8, height = 3.8)

























# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## ---- 2. shared theme ------------------------------------

house <- function() {
  theme(
    plot.title = element_text(face = "bold",
                              size = 12,
                              hjust = 0.5,
                              margin = margin(b = 6)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 10, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 2, 0),
    legend.box.spacing = unit(0.08, "cm"),
    plot.margin = margin(6, 6, 4, 4)
  )
}

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

ACOUNT <- tribble(
  ~endpoint,        ~n,   ~ev,
  "Composite",      1027, 210,
  "30-day death",   1027, 161,
  "CV readmission",  871,  54
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         lab = paste0("n = ", format(n, big.mark = ","),
                      "; ", ev, " events"))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    data = ACOUNT,
    aes(x = endpoint, y = 0.756, label = lab),
    inherit.aes = FALSE,
    size = 2.6, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.765)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Against Acquisition Context"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 3.9)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## Source: data/cxr_strata_ci.csv
##
## Strata differ in size, so n and events are annotated per
## bar rather than once per group. Two lines keeps each
## annotation narrow enough to avoid collision.
##
## Cardiovascular readmission is excluded: 19, 15 and 20
## events per stratum give intervals too wide to interpret.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,   ~lo,    ~hi,    ~n,  ~ev,
  "Composite",    "All portable",  0.6303, 0.5747, 0.6845, 468, 119,
  "Composite",    "Mixed",         0.6462, 0.5585, 0.7333, 184,  47,
  "Composite",    "None portable", 0.6817, 0.5995, 0.7680, 375,  44,
  "30-day death", "All portable",  0.6456, 0.5865, 0.7007, 468, 103,
  "30-day death", "Mixed",         0.6441, 0.5415, 0.7470, 184,  32,
  "30-day death", "None portable", 0.6788, 0.5586, 0.7888, 375,  26
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0("n = ", n, "\n",
                           ev, " events"))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_B,
    width = 0.14, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(y = 0.862, label = nlab),
    position = DODGE_B,
    vjust = 1,
    lineheight = 0.92,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.80, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.865)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Within Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 6.2, height = 3.9)






# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## ---- 2. shared theme ------------------------------------

house <- function() {
  theme(
    plot.title = element_text(face = "bold",
                              size = 12,
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 10, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 6, 0),
    plot.margin = margin(6, 6, 4, 4)
  )
}

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

ACOUNT <- tribble(
  ~endpoint,        ~n,   ~ev,
  "Composite",      1027, 210,
  "30-day death",   1027, 161,
  "CV readmission",  871,  54
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         lab = paste0("n = ", format(n, big.mark = ","),
                      "; ", ev, " events"))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    data = ACOUNT,
    aes(x = endpoint, y = 0.762, label = lab),
    inherit.aes = FALSE,
    size = 2.6, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.78)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Against Acquisition Context"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 4.0)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## Source: data/cxr_strata_ci.csv
##
## Strata differ in size, so n and events are annotated per
## bar rather than once per group. Two lines keeps each
## annotation narrow enough to avoid collision.
##
## Cardiovascular readmission is excluded: 19, 15 and 20
## events per stratum give intervals too wide to interpret.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,   ~lo,    ~hi,    ~n,  ~ev,
  "Composite",    "All portable",  0.6303, 0.5747, 0.6845, 468, 119,
  "Composite",    "Mixed",         0.6462, 0.5585, 0.7333, 184,  47,
  "Composite",    "None portable", 0.6817, 0.5995, 0.7680, 375,  44,
  "30-day death", "All portable",  0.6456, 0.5865, 0.7007, 468, 103,
  "30-day death", "Mixed",         0.6441, 0.5415, 0.7470, 184,  32,
  "30-day death", "None portable", 0.6788, 0.5586, 0.7888, 375,  26
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0("n = ", n, "\n",
                           ev, " events"))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_B,
    width = 0.14, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(y = 0.845, label = nlab),
    position = DODGE_B,
    vjust = 1,
    lineheight = 0.92,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.80, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.87)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Within Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 6.2, height = 4.0)




























# ---------------------------------------------------------
# fig12_cxr_acquisition.R
# Discrimination of the CXR modality against acquisition
# context, and within portability strata
# ---------------------------------------------------------

library(ggplot2)
library(ggpattern)
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

## Okabe-Ito
BLUE   <- "#0072B2"
YELLOW <- "#E69F00"
GREY   <- "#999999"
GREEN  <- "#009E73"
PURPLE <- "#CC79A7"
DARK   <- "#333333"

## ---- 2. shared theme ------------------------------------

house <- function() {
  theme(
    plot.title = element_text(face = "bold",
                              size = 12,
                              hjust = 0.5,
                              margin = margin(b = 6)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 10, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.major.y = element_line(colour = "#E8E8E8",
                                      linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 9, colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 2, 0),
    legend.box.spacing = unit(0.08, "cm"),
    plot.margin = margin(6, 6, 4, 4)
  )
}

## =========================================================
## FIGURE 12a - image content against acquisition context
## Source: data/cxr_metadata_ci.csv
## =========================================================

END <- c("Composite", "30-day death",
         "CV readmission")

SRC <- c("CXR image model", "Film portability",
         "Total film count")

a <- tribble(
  ~endpoint,        ~source,            ~auc,   ~lo,    ~hi,
  "Composite",      "CXR image model",  0.6694, 0.6311, 0.7066,
  "Composite",      "Film portability", 0.5977, 0.5600, 0.6351,
  "Composite",      "Total film count", 0.5994, 0.5547, 0.6432,
  "30-day death",   "CXR image model",  0.6809, 0.6412, 0.7221,
  "30-day death",   "Film portability", 0.6360, 0.5948, 0.6760,
  "30-day death",   "Total film count", 0.6056, 0.5542, 0.6536,
  "CV readmission", "CXR image model",  0.6328, 0.5601, 0.7048,
  "CV readmission", "Film portability", 0.4858, 0.4167, 0.5565,
  "CV readmission", "Total film count", 0.5698, 0.4905, 0.6439
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         source   = factor(source, levels = SRC),
         lab      = sprintf("%.3f", auc))

ACOUNT <- tribble(
  ~endpoint,        ~n,   ~ev,
  "Composite",      1027, 210,
  "30-day death",   1027, 161,
  "CV readmission",  871,  54
) %>%
  mutate(endpoint = factor(endpoint, levels = END),
         lab = paste0("n = ", format(n, big.mark = ",",
                                     trim = TRUE),
                      "; ", ev, " events"))

FILL_A <- setNames(c(BLUE, YELLOW, GREY), SRC)
PAT_A  <- setNames(c("none", "stripe", "crosshatch"), SRC)
ANG_A  <- setNames(c(0, 45, 135), SRC)

DODGE_A <- position_dodge(width = 0.80)

pa <- ggplot(a, aes(x = endpoint, y = auc,
                    group = source)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = source, pattern = source,
        pattern_angle = source),
    position = DODGE_A,
    width = 0.72,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_A,
    width = 0.16, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_A,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    data = ACOUNT,
    aes(x = endpoint, y = 0.756, label = lab),
    inherit.aes = FALSE,
    size = 2.6, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_A, name = NULL) +
  scale_pattern_manual(values = PAT_A, name = NULL) +
  scale_pattern_angle_manual(values = ANG_A,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.75, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.765)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Against Acquisition Context"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pa, "fig12a_cxr_vs_context",
         width = 6.4, height = 3.9)

## =========================================================
## FIGURE 12b - discrimination within portability strata
## Source: data/cxr_strata_ci.csv
##
## Strata differ in size, so n and events are annotated per
## bar rather than once per group, on a single line pinned
## just below the top gridline as in 12a.
##
## Cardiovascular readmission is excluded: 19, 15 and 20
## events per stratum give intervals too wide to interpret.
## =========================================================

STR <- c("All portable", "Mixed", "None portable")

b <- tribble(
  ~endpoint,      ~stratum,        ~auc,   ~lo,    ~hi,    ~n,  ~ev,
  "Composite",    "All portable",  0.6303, 0.5747, 0.6845, 468, 119,
  "Composite",    "Mixed",         0.6462, 0.5585, 0.7333, 184,  47,
  "Composite",    "None portable", 0.6817, 0.5995, 0.7680, 375,  44,
  "30-day death", "All portable",  0.6456, 0.5865, 0.7007, 468, 103,
  "30-day death", "Mixed",         0.6441, 0.5415, 0.7470, 184,  32,
  "30-day death", "None portable", 0.6788, 0.5586, 0.7888, 375,  26
) %>%
  mutate(endpoint = factor(endpoint,
                           levels = c("Composite",
                                      "30-day death")),
         stratum  = factor(stratum, levels = STR),
         lab      = sprintf("%.3f", auc),
         nlab     = paste0(n, "; ", ev, " ev"))

FILL_B <- setNames(c(GREEN, GREY, PURPLE), STR)
PAT_B  <- setNames(c("none", "crosshatch", "stripe"), STR)
ANG_B  <- setNames(c(0, 45, 135), STR)

DODGE_B <- position_dodge(width = 0.76)

pb <- ggplot(b, aes(x = endpoint, y = auc,
                    group = stratum)) +
  geom_hline(yintercept = 0.5, linetype = "22",
             colour = DARK, linewidth = 0.4) +
  geom_col_pattern(
    aes(fill = stratum, pattern = stratum,
        pattern_angle = stratum),
    position = DODGE_B,
    width = 0.68,
    colour = "white", linewidth = 0.3,
    pattern_fill = "white", pattern_colour = "white",
    pattern_density = 0.12, pattern_spacing = 0.018,
    pattern_key_scale_factor = 0.6
  ) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = DODGE_B,
    width = 0.14, linewidth = 0.45,
    colour = DARK
  ) +
  geom_label(
    aes(label = lab),
    position = DODGE_B,
    vjust = 1.25,
    fill = "white", colour = DARK,
    alpha = 0.90, label.size = 0,
    label.r = unit(0.06, "lines"),
    label.padding = unit(0.10, "lines"),
    size = 2.6, fontface = "bold", family = FONT
  ) +
  geom_text(
    aes(y = 0.806, label = nlab),
    position = DODGE_B,
    size = 2.4, colour = GREY, family = FONT
  ) +
  scale_fill_manual(values = FILL_B, name = NULL) +
  scale_pattern_manual(values = PAT_B, name = NULL) +
  scale_pattern_angle_manual(values = ANG_B,
                             name = NULL) +
  scale_y_continuous(breaks = seq(0.40, 0.80, 0.05)) +
  coord_cartesian(ylim = c(0.40, 0.815)) +
  labs(
    title = paste0("Discrimination of the CXR Modality ",
                   "Within Portability Strata"),
    x = NULL,
    y = "AUROC"
  ) +
  guides(fill = guide_legend(nrow = 1),
         pattern = guide_legend(nrow = 1),
         pattern_angle = guide_legend(nrow = 1)) +
  house()

save_fig(pb, "fig12b_cxr_portability_strata",
         width = 6.2, height = 3.9)




























# ---------------------------------------------------------
# fig14_spesi_agebands.R
# Discrimination by age band, fused model against sPESI-6
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
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

BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

OUT_LAB <- c(composite_30d = "Composite Outcome",
             death_30d     = "30-Day Death",
             death_inhosp  = "In-Hospital Death",
             cv_first      = "CV Readmission")

## ---- 2. data --------------------------------------------

ab <- read.csv(file.path(data_dir,
                         "spesi_agebands.csv"),
               stringsAsFactors = FALSE)

cat("\n--- age band gaps ---\n")
print(ab[, c("outcome", "band", "n", "events",
             "model_auc", "spesi_auc", "gap")],
      digits = 3)

al <- ab %>%
  select(outcome, band, n, events,
         model_auc, m_lo, m_hi,
         spesi_auc, s_lo, s_hi) %>%
  pivot_longer(-c(outcome, band, n, events),
               names_to = c("src", ".value"),
               names_pattern =
                 "(model|m|spesi|s)_?(auc|lo|hi)") %>%
  mutate(src = ifelse(src %in% c("model", "m"),
                      "Fused model", "sPESI-6")) %>%
  group_by(outcome, band, src) %>%
  summarise(auc = max(auc, na.rm = TRUE),
            lo = max(lo, na.rm = TRUE),
            hi = max(hi, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(
    outcome = factor(outcome, levels = names(OUT_LAB),
                     labels = OUT_LAB),
    band = factor(band, levels = c("<50", "50-64",
                                   "65-79", "80+")),
    src = factor(src, levels = c("Fused model",
                                 "sPESI-6")))

## ---- 3. plot --------------------------------------------

fB <- ggplot(al, aes(x = band, y = auc, colour = src,
                     shape = src, group = src)) +
  geom_hline(yintercept = 0.5, colour = DARK,
             linewidth = 0.4, linetype = "22") +
  geom_line(linewidth = 0.7, alpha = 0.6) +
  geom_errorbar(aes(ymin = lo, ymax = hi),
                width = 0.12, linewidth = 0.4,
                position = position_dodge(width = 0.24)) +
  geom_point(size = 2.6,
             position = position_dodge(width = 0.24)) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(
    values = c("Fused model" = BLUE,
               "sPESI-6" = ORANGE), name = NULL) +
  scale_shape_manual(values = c(16, 15), name = NULL) +
  labs(x = "Age band", y = "AUROC",
       title = paste("Discrimination by Age Band for the",
                     "Fused Model and sPESI-6")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    strip.text = element_text(size = 10.5,
                              face = "bold",
                              colour = DARK,
                              margin = margin(b = 4)),
    axis.title.x = element_text(size = 12,
                                colour = DARK,
                                margin = margin(t = 2)),
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 9.5, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.spacing = unit(0.7, "lines"),
    legend.position = "bottom",
    legend.justification = "center",
    legend.text = element_text(size = 9.5,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(2, 0, 0, 0),
    legend.box.spacing = unit(0.10, "cm"),
    plot.margin = margin(6, 6, 4, 4))

print(fB)
save_fig(fB, "fig14_spesi_agebands", 10.4, 4.4)



























# ---------------------------------------------------------
# fig14_subgroups.R
# Discrimination by age band and by sex, fused model
# against sPESI-6
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

BLUE   <- "#0072B2"
ORANGE <- "#D55E00"
DARK   <- "#333333"

PAL <- c("Fused model" = BLUE,
         "sPESI-6"     = ORANGE)

OUT_LAB <- c(composite_30d = "Composite Outcome",
             death_30d     = "30-Day Death",
             death_inhosp  = "In-Hospital Death",
             cv_first      = "CV Readmission")

YLIM <- c(0.45, 0.98)

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
    pd %>% select(outcome, level, n, ev,
                  src, auc, lo, hi)),
    digits = 3)
  ggplot(pd, aes(x = level, y = auc, colour = src,
                 shape = src, group = src)) +
    geom_hline(yintercept = 0.5, colour = DARK,
               linewidth = 0.4, linetype = "22") +
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
    theme(
      plot.title = element_text(size = 12,
                                face = "bold",
                                hjust = 0.5,
                                margin = margin(b = 10)),
      plot.title.position = "plot",
      strip.text = element_text(size = 10.5,
                                face = "bold",
                                colour = DARK,
                                margin = margin(b = 4)),
      axis.title.x = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(t = 2)),
      axis.title.y = element_text(size = 12,
                                  colour = DARK,
                                  margin = margin(r = 2)),
      axis.text = element_text(size = 9.5,
                               colour = DARK),
      axis.ticks.length = unit(0.06, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(
        colour = "#E8E8E8", linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.spacing = unit(0.7, "lines"),
      legend.position = "bottom",
      legend.justification = "center",
      legend.text = element_text(size = 9.5,
                                 colour = DARK,
                                 margin = margin(l = 3,
                                                 r = 16)),
      legend.key.size = unit(0.40, "cm"),
      legend.margin = margin(2, 0, 0, 0),
      legend.box.spacing = unit(0.10, "cm"),
      plot.margin = margin(6, 8, 4, 6))
}

## ---- 3. age bands ---------------------------------------

AGE_VARS <- c("band", "age", "age_band")
age <- s %>% filter(var %in% AGE_VARS)
stopifnot(nrow(age) > 0)

lv_age <- intersect(c("<50", "50-64", "65-79", "80+"),
                    unique(age$level))
f_age <- make_fig(
  age, lv_age, "Age band",
  paste("Discrimination by Age Band for the Fused",
        "Model and sPESI-6"))

cat("\nwriting figure 14a...\n")
print(system.time(
  save_fig(f_age, "fig14a_age_bands", 10.4, 4.4)))

## ---- 4. sex ---------------------------------------------

SEX_VARS <- c("sex", "gender")
sex <- s %>% filter(var %in% SEX_VARS)

if (nrow(sex) == 0) {
  message("no sex rows found - figure 14b skipped")
} else {
  lv_sex <- unique(sex$level)
  f_sex <- make_fig(
    sex, lv_sex, "Sex",
    paste("Discrimination by Sex for the Fused",
          "Model and sPESI-6"))
  cat("\nwriting figure 14b...\n")
  print(system.time(
    save_fig(f_sex, "fig14b_sex", 10.4, 4.4)))
}
































# ---------------------------------------------------------
# fig15_stability_intervals.R
# Confidence interval width by modality and endpoint
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

DARK <- "#333333"
GREY <- "#7F7F7F"

OUT_LAB <- c(composite_30d = "Composite Outcome",
             death_30d = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             cv_first = "CV Readmission")

MOD_MAP <- c("Structured EHR"    = "EHR",
             "CTPA report"       = "CTPA",
             "Electrocardiogram" = "ECG",
             "Chest radiograph"  = "CXR")

MOD_ORD <- c("EHR", "CTPA", "ECG", "CXR")

## Okabe-Ito
MOD_PAL <- c("EHR"  = "#0072B2",
             "CTPA" = "#009E73",
             "ECG"  = "#E69F00",
             "CXR"  = "#CC79A7")

## Shape is redundant with colour for greyscale.
MOD_SHP <- c("EHR"  = 16,
             "CTPA" = 17,
             "ECG"  = 15,
             "CXR"  = 18)

## ---- 2. data --------------------------------------------

iv <- read.csv(file.path(data_dir,
                         "stability_intervals.csv"),
               stringsAsFactors = FALSE)

cat("\n--- interval width vs events ---\n")
print(iv[order(iv$events),
         c("arm", "outcome", "events", "auc",
           "lo", "hi", "width")], digits = 3)
cat("corr(log events, width):",
    sprintf("%.3f", cor(log(iv$events), iv$width)), "\n")

pa <- iv %>%
  mutate(modality = MOD_MAP[arm],
         modality = factor(modality,
                           levels = MOD_ORD),
         outcome = factor(outcome,
                          levels = names(OUT_LAB),
                          labels = OUT_LAB)) %>%
  filter(!is.na(modality))

miss <- setdiff(unique(iv$arm), names(MOD_MAP))
if (length(miss) > 0) {
  message("unmapped modality label: ",
          paste(miss, collapse = ", "))
}

## ---- 3. plot --------------------------------------------

f1 <- ggplot(pa, aes(x = auc, y = modality,
                     colour = modality)) +
  geom_vline(xintercept = 0.5, colour = DARK,
             linewidth = 0.4, linetype = "22") +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 height = 0.20, linewidth = 0.55) +
  geom_point(aes(shape = modality), size = 2.6) +
  geom_text(aes(x = hi, label = paste0(events, " ev")),
            hjust = -0.20, size = 2.6, colour = GREY,
            family = FONT) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = MOD_PAL,
                      guide = "none") +
  scale_shape_manual(values = MOD_SHP,
                     guide = "none") +
  scale_x_continuous(
    limits = c(0.44, 1.00),
    breaks = seq(0.5, 0.9, by = 0.2),
    labels = function(x) sprintf("%.1f", x)) +
  scale_y_discrete(limits = rev(MOD_ORD)) +
  labs(x = "AUROC (95% bootstrap interval)", y = NULL,
       title = paste("Interval Width by Modality and",
                     "Endpoint")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    strip.text = element_text(size = 10.5,
                              face = "bold",
                              colour = DARK,
                              margin = margin(b = 4)),
    axis.title.x = element_text(size = 12,
                                colour = DARK,
                                margin = margin(t = 2)),
    axis.text.y = element_text(size = 11,
                               colour = DARK),
    axis.text.x = element_text(size = 9.5,
                               colour = GREY),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    panel.spacing = unit(0.7, "lines"),
    plot.margin = margin(6, 12, 4, 6))

print(f1)
save_fig(f1, "fig15_stability_intervals", 10.6, 3.4)




































# ---------------------------------------------------------
# fig15_interval_width.R
# Confidence interval width against event count
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

DARK <- "#333333"
GREY <- "#7F7F7F"

OUT_LAB <- c(composite_30d = "Composite",
             death_30d = "30-day death",
             death_30d_inhosp = "In-hospital death",
             cv_first = "CV readmission")

MOD_MAP <- c("Structured EHR"    = "EHR",
             "CTPA report"       = "CTPA",
             "Electrocardiogram" = "ECG",
             "Chest radiograph"  = "CXR")

MOD_ORD <- c("EHR", "CTPA", "ECG", "CXR")

## Okabe-Ito
MOD_PAL <- c("EHR"  = "#0072B2",
             "CTPA" = "#009E73",
             "ECG"  = "#E69F00",
             "CXR"  = "#CC79A7")

MOD_SHP <- c("EHR"  = 16,
             "CTPA" = 17,
             "ECG"  = 15,
             "CXR"  = 18)

## ---- 2. data --------------------------------------------

iv <- read.csv(file.path(data_dir,
                         "stability_intervals.csv"),
               stringsAsFactors = FALSE)

d <- iv %>%
  mutate(modality = MOD_MAP[arm],
         modality = factor(modality, levels = MOD_ORD),
         endpoint = OUT_LAB[outcome]) %>%
  filter(!is.na(modality))

miss <- setdiff(unique(iv$arm), names(MOD_MAP))
if (length(miss) > 0) {
  message("unmapped modality label: ",
          paste(miss, collapse = ", "))
}

RHO <- cor(log(d$events), d$width)
cat("\ncells:", nrow(d), "\n")
cat("corr(log events, width):",
    sprintf("%.3f", RHO), "\n\n")
print(d[order(-d$events),
        c("modality", "endpoint", "events", "width")],
      digits = 3, row.names = FALSE)

KEY <- d %>%
  filter(events == max(events) |
         events == min(events) |
         width == max(width) |
         width == min(width) |
         (modality == "EHR" &
            endpoint %in% c("Composite",
                            "CV readmission"))) %>%
  mutate(lab = paste0(modality, ", ", endpoint))

## ---- 3. plot --------------------------------------------

f <- ggplot(d, aes(x = events, y = width)) +
  geom_smooth(method = "lm", formula = y ~ x,
              se = FALSE, colour = GREY,
              linewidth = 0.5, linetype = "22") +
  geom_point(aes(colour = modality, shape = modality),
             size = 3.0) +
  geom_text(data = KEY,
            aes(label = lab, colour = modality),
            hjust = -0.14, size = 2.8,
            family = FONT, show.legend = FALSE) +
  annotate("text", x = 58, y = 0.030,
           label = sprintf("r = %.2f", RHO),
           hjust = 0, size = 3.4, colour = DARK,
           fontface = "bold", family = FONT) +
  scale_colour_manual(values = MOD_PAL, name = NULL) +
  scale_shape_manual(values = MOD_SHP, name = NULL) +
  scale_x_log10(
    limits = c(50, 1400),
    breaks = c(50, 100, 200, 400, 800),
    expand = expansion(mult = c(0.02, 0.02))) +
  scale_y_continuous(
    limits = c(0.025, 0.145),
    breaks = seq(0.04, 0.14, 0.02),
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = c(0.02, 0.02))) +
  labs(
    x = "Positive events in the evaluation cell (log scale)",
    y = "Width of the 95% interval",
    title = paste("Interval Width Against Event Count",
                  "Across Sixteen Evaluation Cells")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.x = element_text(size = 12,
                                colour = DARK,
                                margin = margin(t = 2)),
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text = element_text(size = 9.5, colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 9.5,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 16)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(0, 0, 2, 0),
    legend.box.spacing = unit(0.08, "cm"),
    plot.margin = margin(6, 14, 4, 4))

print(f)
save_fig(f, "fig15_interval_width", 6.8, 4.6)











































# ---------------------------------------------------------
# fig16_stability_folds.R
# Fold-level discrimination within the CXR modality
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

PURPLE <- "#CC79A7"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

OUT_LAB <- c(composite_30d = "Composite",
             death_30d = "30-day death",
             death_30d_inhosp = "In-hospital death",
             cv_first = "CV readmission")

EV_FOLD <- c("Composite" = 53,
             "30-day death" = 39,
             "In-hospital death" = 21,
             "CV readmission" = 18)

## ---- 2. data --------------------------------------------

fl <- read.csv(file.path(data_dir,
                         "stability_folds.csv"),
               stringsAsFactors = FALSE)

cat("\n--- fold spread ---\n")
print(fl %>%
        group_by(arm, outcome) %>%
        summarise(folds = n(),
                  min = min(auc), max = max(auc),
                  range = max(auc) - min(auc),
                  sd = sd(auc), .groups = "drop") %>%
        as.data.frame(), digits = 3)

pb <- fl %>%
  mutate(endpoint = OUT_LAB[outcome]) %>%
  filter(!is.na(endpoint))

## order by events per fold, descending
lv <- names(sort(EV_FOLD[unique(pb$endpoint)],
                 decreasing = TRUE))
pb$endpoint <- factor(pb$endpoint, levels = lv)

SUMM <- pb %>%
  group_by(endpoint) %>%
  summarise(m = mean(auc),
            lo = min(auc),
            hi = max(auc),
            sd = sd(auc),
            .groups = "drop") %>%
  mutate(sdlab = sprintf("SD %.3f", sd),
         evlab = paste0(EV_FOLD[as.character(endpoint)],
                        " ev/fold"))

ylo <- floor(min(pb$auc) * 20) / 20 - 0.01
yhi <- ceiling(max(pb$auc) * 20) / 20 + 0.05

## ---- 3. plot --------------------------------------------

f2 <- ggplot(pb, aes(x = endpoint, y = auc)) +
  geom_linerange(data = SUMM,
                 aes(x = endpoint, ymin = lo, ymax = hi),
                 linewidth = 5.2, colour = "#F0DCE7",
                 inherit.aes = FALSE) +
  geom_crossbar(data = SUMM,
                aes(x = endpoint, y = m,
                    ymin = m, ymax = m),
                width = 0.34, linewidth = 0.5,
                colour = DARK,
                inherit.aes = FALSE) +
  geom_point(size = 2.6, alpha = 0.9,
             colour = PURPLE,
             position = position_jitter(
               width = 0.05, height = 0, seed = 42)) +
  geom_text(data = SUMM,
            aes(x = endpoint, y = hi + 0.016,
                label = sdlab),
            size = 2.9, colour = DARK,
            fontface = "bold", family = FONT,
            inherit.aes = FALSE) +
  geom_text(data = SUMM,
            aes(x = endpoint, y = hi + 0.036,
                label = evlab),
            size = 2.6, colour = GREY,
            family = FONT, inherit.aes = FALSE) +
  scale_y_continuous(
    limits = c(ylo, yhi),
    breaks = seq(0.55, 0.80, by = 0.05),
    labels = function(x) sprintf("%.2f", x)) +
  labs(x = NULL, y = "AUROC",
       title = paste("Fold-Level Discrimination in the",
                     "CXR Modality, by Endpoint")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text.x = element_text(size = 11,
                               colour = DARK),
    axis.text.y = element_text(size = 9.5,
                               colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    plot.margin = margin(6, 10, 4, 4))

print(f2)
save_fig(f2, "fig16_stability_folds", 6.4, 4.0)











































# ---------------------------------------------------------
# fig16_stability_folds.R
# Fold-level discrimination within the CXR modality
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

PURPLE <- "#CC79A7"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

OUT_LAB <- c(composite_30d = "Composite",
             death_30d = "30-day death",
             death_30d_inhosp = "In-hospital death",
             cv_first = "CV readmission")

EV_FOLD <- c("Composite" = 53,
             "30-day death" = 39,
             "In-hospital death" = 21,
             "CV readmission" = 18)

## ---- 2. data --------------------------------------------

fl <- read.csv(file.path(data_dir,
                         "stability_folds.csv"),
               stringsAsFactors = FALSE)

cat("\n--- fold spread ---\n")
print(fl %>%
        group_by(arm, outcome) %>%
        summarise(folds = n(),
                  min = min(auc), max = max(auc),
                  range = max(auc) - min(auc),
                  sd = sd(auc), .groups = "drop") %>%
        as.data.frame(), digits = 3)

pb <- fl %>%
  mutate(endpoint = OUT_LAB[outcome]) %>%
  filter(!is.na(endpoint))

no_ev <- setdiff(unique(pb$endpoint), names(EV_FOLD))
if (length(no_ev) > 0) {
  message("no events-per-fold set for: ",
          paste(no_ev, collapse = ", "),
          " - dropped")
  pb <- pb %>% filter(endpoint %in% names(EV_FOLD))
}

cat("\nendpoints plotted:",
    paste(unique(pb$endpoint), collapse = ", "), "\n")

## order by events per fold, descending
lv <- names(sort(EV_FOLD[unique(pb$endpoint)],
                 decreasing = TRUE))
pb$endpoint <- factor(pb$endpoint, levels = lv)

SUMM <- pb %>%
  group_by(endpoint) %>%
  summarise(m = mean(auc),
            lo = min(auc),
            hi = max(auc),
            sd = sd(auc),
            .groups = "drop") %>%
  mutate(sdlab = sprintf("SD %.3f", sd),
         evlab = paste0(EV_FOLD[as.character(endpoint)],
                        " ev/fold"))

ylo <- min(pb$auc) - 0.018
yhi <- max(pb$auc) + 0.055

## ---- 3. plot --------------------------------------------

f2 <- ggplot(pb, aes(x = endpoint, y = auc)) +
  geom_linerange(data = SUMM,
                 aes(x = endpoint, ymin = lo, ymax = hi),
                 linewidth = 5.2, colour = "#F0DCE7",
                 inherit.aes = FALSE) +
  geom_crossbar(data = SUMM,
                aes(x = endpoint, y = m,
                    ymin = m, ymax = m),
                width = 0.14, linewidth = 0.5,
                colour = DARK,
                inherit.aes = FALSE) +
  geom_point(size = 2.6, alpha = 0.9,
             colour = PURPLE) +
  geom_text(data = SUMM,
            aes(x = endpoint, y = hi + 0.018,
                label = sdlab),
            size = 2.9, colour = DARK,
            fontface = "bold", family = FONT,
            inherit.aes = FALSE) +
  geom_text(data = SUMM,
            aes(x = endpoint, y = hi + 0.038,
                label = evlab),
            size = 2.6, colour = GREY,
            family = FONT, inherit.aes = FALSE) +
  scale_y_continuous(
    limits = c(ylo, yhi),
    breaks = seq(0.55, 0.80, by = 0.05),
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = c(0, 0))) +
  labs(x = NULL, y = "AUROC",
       title = paste("Fold-Level Discrimination in the",
                     "CXR Modality, by Endpoint")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text.x = element_text(size = 11,
                               colour = DARK),
    axis.text.y = element_text(size = 9.5,
                               colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    plot.margin = margin(6, 10, 4, 4))

print(f2)
save_fig(f2, "fig16_stability_folds", 6.4, 4.0)






































# ---------------------------------------------------------
# fig16_stability_folds.R
# Fold-level discrimination within the CXR modality
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

PURPLE <- "#CC79A7"
DARK   <- "#333333"
GREY   <- "#7F7F7F"

OUT_LAB <- c(composite_30d = "Composite",
             death_30d = "30-day death",
             death_30d_inhosp = "In-hospital death",
             cv_first = "CV readmission")

EV_FOLD <- c("Composite" = 53,
             "30-day death" = 39,
             "In-hospital death" = 21,
             "CV readmission" = 18)

## ---- 2. data --------------------------------------------

fl <- read.csv(file.path(data_dir,
                         "stability_folds.csv"),
               stringsAsFactors = FALSE)

cat("\n--- fold spread ---\n")
print(fl %>%
        group_by(arm, outcome) %>%
        summarise(folds = n(),
                  min = min(auc), max = max(auc),
                  range = max(auc) - min(auc),
                  sd = sd(auc), .groups = "drop") %>%
        as.data.frame(), digits = 3)

pb <- fl %>%
  mutate(endpoint = OUT_LAB[outcome]) %>%
  filter(!is.na(endpoint))

no_ev <- setdiff(unique(pb$endpoint), names(EV_FOLD))
if (length(no_ev) > 0) {
  message("no events-per-fold set for: ",
          paste(no_ev, collapse = ", "),
          " - dropped")
  pb <- pb %>% filter(endpoint %in% names(EV_FOLD))
}

cat("\nendpoints plotted:",
    paste(unique(pb$endpoint), collapse = ", "), "\n")

## order by events per fold, descending
lv <- names(sort(EV_FOLD[unique(pb$endpoint)],
                 decreasing = TRUE))
pb$endpoint <- factor(pb$endpoint, levels = lv)

SUMM <- pb %>%
  group_by(endpoint) %>%
  summarise(m = mean(auc),
            lo = min(auc),
            hi = max(auc),
            sd = sd(auc),
            .groups = "drop") %>%
  mutate(sdlab = sprintf("SD %.3f", sd),
         evlab = paste0(EV_FOLD[as.character(endpoint)],
                        " ev/fold"))

ylo <- min(pb$auc) - 0.020
yhi <- max(pb$auc) + 0.058

## ---- 3. plot --------------------------------------------

f2 <- ggplot(pb, aes(x = endpoint, y = auc)) +
  geom_linerange(data = SUMM,
                 aes(x = endpoint, ymin = lo, ymax = hi),
                 linewidth = 5.2, colour = "#F0DCE7",
                 inherit.aes = FALSE) +
  geom_crossbar(data = SUMM,
                aes(x = endpoint, y = m,
                    ymin = m, ymax = m),
                width = 0.14, linewidth = 0.5,
                colour = DARK,
                inherit.aes = FALSE) +
  geom_point(size = 2.6, alpha = 0.9,
             colour = PURPLE) +
  geom_text(data = SUMM,
            aes(x = endpoint, y = hi + 0.018,
                label = sdlab),
            size = 2.9, colour = DARK,
            fontface = "bold", family = FONT,
            inherit.aes = FALSE) +
  geom_text(data = SUMM,
            aes(x = endpoint, y = hi + 0.038,
                label = evlab),
            size = 2.6, colour = GREY,
            family = FONT, inherit.aes = FALSE) +
  scale_y_continuous(
    breaks = seq(0.55, 0.80, by = 0.05),
    labels = function(x) sprintf("%.2f", x)) +
  coord_cartesian(ylim = c(ylo, yhi)) +
  labs(x = NULL, y = "AUROC",
       title = paste("Fold-Level Discrimination in the",
                     "CXR Modality, by Endpoint")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text.x = element_text(size = 11,
                               colour = DARK),
    axis.text.y = element_text(size = 9.5,
                               colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    plot.margin = margin(6, 10, 4, 4))

print(f2)
save_fig(f2, "fig16_stability_folds", 6.4, 4.0)








































# ---------------------------------------------------------
# fig18_fusion_rules.R
# Two-modality combination rules against the weighted
# average
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(scales)
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

DARK <- "#333333"
GREY <- "#7F7F7F"

## ---- 2. two-modality rules only -------------------------

d <- tribble(
  ~family, ~rule, ~outcome, ~diff, ~lo, ~hi,

  # --- reference point, not a combination rule ---
  "Reference", "Best single modality",
  "Composite", -0.0089, NA, NA,
  "Reference", "Best single modality",
  "30-day death", -0.0126, NA, NA,

  # --- fixed and fitted late rules (ref WMEAN2) ---
  "Fixed and fitted", "Equal-weight average",
  "Composite", -0.0100, NA, NA,
  "Fixed and fitted", "Equal-weight average",
  "30-day death", -0.0121, NA, NA,
  "Fixed and fitted", "Logistic stack",
  "Composite", -0.0003, NA, NA,
  "Fixed and fitted", "Logistic stack",
  "30-day death", -0.0013, NA, NA,

  # --- gated and adaptive (ref WMEAN2, earlier run) ---
  "Gated and adaptive", "Residual",
  "Composite", -0.0010, -0.0017, -0.0003,
  "Gated and adaptive", "Residual",
  "30-day death", -0.0006, -0.0018, 0.0006,
  "Gated and adaptive", "Gated on confidence",
  "Composite", -0.0017, -0.0043, 0.0008,
  "Gated and adaptive", "Gated on confidence",
  "30-day death", -0.0018, -0.0038, 0.0002,
  "Gated and adaptive", "Gated on recording count",
  "Composite", -0.0016, -0.0030, -0.0003,
  "Gated and adaptive", "Gated on recording count",
  "30-day death", -0.0012, -0.0025, 0.0002,
  "Gated and adaptive", "Gated on delay",
  "Composite", -0.0044, -0.0074, -0.0015,
  "Gated and adaptive", "Gated on delay",
  "30-day death", -0.0013, -0.0034, 0.0009,
  "Gated and adaptive", "Gated on all three",
  "Composite", -0.0061, -0.0103, -0.0022,
  "Gated and adaptive", "Gated on all three",
  "30-day death", -0.0034, -0.0062, -0.0008,

  # --- architecture (ref late, MIMIC-trained EHR) ---
  "Architecture", "Early concatenation",
  "Composite", -0.0008, -0.0055, 0.0037,
  "Architecture", "Early concatenation",
  "30-day death", -0.0021, -0.0072, 0.0028
)

FAM <- c("Reference", "Fixed and fitted",
         "Gated and adaptive", "Architecture")

## Okabe-Ito
PAL <- c("Reference"          = "#666666",
         "Fixed and fitted"   = "#0072B2",
         "Gated and adaptive" = "#D55E00",
         "Architecture"       = "#009E73")

## ---- 3. order rules within family ------------------------

ord <- d %>%
  filter(outcome == "Composite") %>%
  mutate(family = factor(family, levels = FAM)) %>%
  arrange(family, diff) %>%
  pull(rule)

pd <- d %>%
  mutate(family = factor(family, levels = FAM),
         rule = factor(rule, levels = rev(ord)),
         outcome = factor(outcome,
                          levels = c("Composite",
                                     "30-day death")),
         sig = !is.na(lo) & (lo > 0 | hi < 0),
         has_ci = !is.na(lo))

XLIM <- c(-0.016, 0.006)

n_out <- sum(pd$diff < XLIM[1] | pd$diff > XLIM[2],
             na.rm = TRUE)
cat("\npoints outside axis limits:", n_out, "\n")
if (n_out > 0) {
  message("WARNING: squished points are not marked ",
          "on the figure - widen XLIM")
}
cat("rules plotted:",
    length(unique(pd$rule)), "\n")
cat("rules without an interval:",
    paste(unique(pd$rule[!pd$has_ci]),
          collapse = ", "), "\n\n")
print(as.data.frame(
  pd %>% filter(outcome == "Composite") %>%
    select(family, rule, diff, has_ci, sig)),
  digits = 3)

SEP <- length(levels(pd$rule)) - 0.5

## ---- 4. plot and save -----------------------------------

f <- ggplot(pd, aes(x = diff, y = rule,
                    colour = family)) +
  geom_vline(xintercept = 0, colour = DARK,
             linewidth = 0.5) +
  geom_hline(yintercept = SEP, colour = "#DDDDDD",
             linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 height = 0, linewidth = 0.5,
                 na.rm = TRUE) +
  geom_point(aes(shape = sig), size = 2.6,
             fill = "white", stroke = 0.8) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, name = NULL) +
  scale_shape_manual(values = c("TRUE" = 16,
                                "FALSE" = 21),
                     guide = "none") +
  scale_x_continuous(
    limits = XLIM, oob = squish,
    breaks = seq(-0.015, 0.005, by = 0.005),
    labels = function(x) sprintf("%+.3f", x)) +
  labs(x = paste("Change in AUROC relative to the",
                 "weighted average"),
       y = NULL,
       title = paste("Combination Rules Against the",
                     "Weighted Average, Two Modalities")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    strip.text = element_text(size = 10.5,
                              face = "bold",
                              colour = DARK,
                              margin = margin(b = 4)),
    axis.title.x = element_text(size = 12,
                                colour = DARK,
                                margin = margin(t = 2)),
    axis.text.y = element_text(size = 10,
                               colour = DARK),
    axis.text.x = element_text(size = 9.5,
                               colour = GREY),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    panel.spacing = unit(0.7, "lines"),
    legend.position = "bottom",
    legend.justification = "center",
    legend.text = element_text(size = 9.5,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 14)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(2, 0, 0, 0),
    legend.box.spacing = unit(0.10, "cm"),
    plot.margin = margin(6, 10, 4, 4))

print(f)
save_fig(f, "fig18_fusion_rules", 8.2, 4.4)





























# ---------------------------------------------------------
# fig18_fusion_rules.R
# Two-modality combination rules against the weighted
# average
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tibble)
library(scales)
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

DARK <- "#333333"
GREY <- "#7F7F7F"

## ---- 2. two-modality rules only -------------------------

d <- tribble(
  ~family, ~rule, ~outcome, ~diff, ~lo, ~hi,

  # --- reference point, not a combination rule ---
  "Reference", "Best single modality",
  "Composite", -0.0089, NA, NA,
  "Reference", "Best single modality",
  "30-day death", -0.0126, NA, NA,

  # --- fixed and fitted late rules (ref WMEAN2) ---
  "Fixed and fitted", "Equal-weight average",
  "Composite", -0.0100, NA, NA,
  "Fixed and fitted", "Equal-weight average",
  "30-day death", -0.0121, NA, NA,
  "Fixed and fitted", "Logistic stack",
  "Composite", -0.0003, NA, NA,
  "Fixed and fitted", "Logistic stack",
  "30-day death", -0.0013, NA, NA,

  # --- gated and adaptive (ref WMEAN2, earlier run) ---
  "Gated and adaptive", "Residual",
  "Composite", -0.0010, -0.0017, -0.0003,
  "Gated and adaptive", "Residual",
  "30-day death", -0.0006, -0.0018, 0.0006,
  "Gated and adaptive", "Gated on confidence",
  "Composite", -0.0017, -0.0043, 0.0008,
  "Gated and adaptive", "Gated on confidence",
  "30-day death", -0.0018, -0.0038, 0.0002,
  "Gated and adaptive", "Gated on recording count",
  "Composite", -0.0016, -0.0030, -0.0003,
  "Gated and adaptive", "Gated on recording count",
  "30-day death", -0.0012, -0.0025, 0.0002,
  "Gated and adaptive", "Gated on delay",
  "Composite", -0.0044, -0.0074, -0.0015,
  "Gated and adaptive", "Gated on delay",
  "30-day death", -0.0013, -0.0034, 0.0009,
  "Gated and adaptive", "Gated on all three",
  "Composite", -0.0061, -0.0103, -0.0022,
  "Gated and adaptive", "Gated on all three",
  "30-day death", -0.0034, -0.0062, -0.0008,

  # --- architecture (ref late, MIMIC-trained EHR) ---
  "Architecture", "Early concatenation",
  "Composite", -0.0008, -0.0055, 0.0037,
  "Architecture", "Early concatenation",
  "30-day death", -0.0021, -0.0072, 0.0028
)

FAM <- c("Reference", "Fixed and fitted",
         "Gated and adaptive", "Architecture")

## Okabe-Ito
PAL <- c("Reference"          = "#666666",
         "Fixed and fitted"   = "#0072B2",
         "Gated and adaptive" = "#D55E00",
         "Architecture"       = "#009E73")

STATE <- c("Interval excludes zero",
           "Interval spans zero",
           "Not formally tested")

SHP <- c("Interval excludes zero" = 16,
         "Interval spans zero"    = 21,
         "Not formally tested"    = 4)

## ---- 3. order rules within family ------------------------

ord <- d %>%
  filter(outcome == "Composite") %>%
  mutate(family = factor(family, levels = FAM)) %>%
  arrange(family, diff) %>%
  pull(rule)

pd <- d %>%
  mutate(family = factor(family, levels = FAM),
         rule = factor(rule, levels = rev(ord)),
         outcome = factor(outcome,
                          levels = c("Composite",
                                     "30-day death")),
         state = case_when(
           is.na(lo) ~ "Not formally tested",
           lo > 0 | hi < 0 ~ "Interval excludes zero",
           TRUE ~ "Interval spans zero"),
         state = factor(state, levels = STATE))

XLIM <- c(-0.016, 0.006)

n_out <- sum(pd$diff < XLIM[1] | pd$diff > XLIM[2],
             na.rm = TRUE)
cat("\npoints outside axis limits:", n_out, "\n")
if (n_out > 0) {
  message("WARNING: squished points are not marked ",
          "on the figure - widen XLIM")
}
cat("rules plotted:",
    length(unique(pd$rule)), "\n\n")
print(as.data.frame(
  pd %>% filter(outcome == "Composite") %>%
    select(family, rule, diff, state)), digits = 3)

SEP <- length(levels(pd$rule)) - 0.5

## ---- 4. plot and save -----------------------------------

f <- ggplot(pd, aes(x = diff, y = rule,
                    colour = family)) +
  geom_vline(xintercept = 0, colour = DARK,
             linewidth = 0.5) +
  geom_hline(yintercept = SEP, colour = "#DDDDDD",
             linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 height = 0, linewidth = 0.5,
                 na.rm = TRUE) +
  geom_point(aes(shape = state), size = 2.6,
             fill = "white", stroke = 0.9) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, name = NULL) +
  scale_shape_manual(values = SHP, name = NULL,
                     drop = FALSE) +
  scale_x_continuous(
    limits = XLIM, oob = squish,
    breaks = seq(-0.015, 0.005, by = 0.005),
    labels = function(x) sprintf("%+.3f", x)) +
  labs(x = paste("Change in AUROC relative to the",
                 "weighted average"),
       y = NULL,
       title = paste("Combination Rules Against the",
                     "Weighted Average, Two Modalities")) +
  guides(
    colour = "none",
    shape = guide_legend(
      nrow = 1,
      override.aes = list(colour = DARK, size = 2.6))) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    strip.text = element_text(size = 10.5,
                              face = "bold",
                              colour = DARK,
                              margin = margin(b = 4)),
    axis.title.x = element_text(size = 12,
                                colour = DARK,
                                margin = margin(t = 2)),
    axis.text.y = element_text(size = 10,
                               colour = DARK),
    axis.text.x = element_text(size = 9.5,
                               colour = GREY),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    panel.spacing = unit(0.7, "lines"),
    legend.position = "bottom",
    legend.justification = "center",
    legend.box = "vertical",
    legend.text = element_text(size = 9.5,
                               colour = DARK,
                               margin = margin(l = 3,
                                               r = 14)),
    legend.key.size = unit(0.40, "cm"),
    legend.margin = margin(2, 0, 0, 0),
    legend.box.spacing = unit(0.10, "cm"),
    plot.margin = margin(6, 10, 4, 4))

print(f)
save_fig(f, "fig18_fusion_rules", 8.2, 4.8)










































# ---------------------------------------------------------
# fig19_twoway_weights.R
# Fold-selected weight on the EHR modality, two-modality
# fusion
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

BLUE <- "#0072B2"
DARK <- "#333333"
GREY <- "#7F7F7F"

OUT_ORD <- c("composite_30d", "death_30d", "cv_first")

OUT_LAB <- c(composite_30d = "Composite outcome",
             death_30d = "30-day death",
             cv_first = "CV readmission")

GAP <- c("Composite outcome" = 0.0793,
         "30-day death"      = 0.1003,
         "CV readmission"    = 0.1073)

## ---- 2. data --------------------------------------------

wt <- read.csv(file.path(data_dir,
                         "twoway_weights.csv"),
               stringsAsFactors = FALSE)

cat("\n--- weights (fold 0 = oracle) ---\n")
print(wt, digits = 2)

keep_w <- intersect(OUT_ORD, unique(wt$outcome))

folds <- wt %>%
  filter(fold > 0, outcome %in% keep_w) %>%
  mutate(outcome = factor(outcome, levels = keep_w,
                          labels = OUT_LAB[keep_w]))

orc <- wt %>%
  filter(fold == 0, outcome %in% keep_w) %>%
  mutate(outcome = factor(outcome, levels = keep_w,
                          labels = OUT_LAB[keep_w]))

ANN <- folds %>%
  group_by(outcome) %>%
  summarise(top = max(w_ehr), .groups = "drop") %>%
  mutate(lab = sprintf("gap %.3f",
                       GAP[as.character(outcome)]))

cat("\n--- weight against gap ---\n")
print(folds %>%
        group_by(outcome) %>%
        summarise(folds = n(),
                  modal = names(sort(table(w_ehr),
                                     decreasing = TRUE))[1],
                  min = min(w_ehr), max = max(w_ehr),
                  .groups = "drop") %>%
        mutate(gap = GAP[as.character(outcome)]) %>%
        as.data.frame(), digits = 3)

ylo <- min(c(folds$w_ehr, orc$w_ehr)) - 0.045
yhi <- min(1.0, max(c(folds$w_ehr, orc$w_ehr)) + 0.075)

## ---- 3. plot --------------------------------------------

f2 <- ggplot(folds, aes(x = outcome, y = w_ehr)) +
  geom_errorbar(data = orc,
                aes(x = outcome, ymin = w_ehr,
                    ymax = w_ehr),
                width = 0.30, linewidth = 0.5,
                colour = DARK,
                inherit.aes = FALSE) +
  geom_beeswarm(size = 2.8, colour = BLUE,
                alpha = 0.9, cex = 2.6,
                method = "center") +
  geom_text(data = ANN,
            aes(x = outcome, y = top + 0.030,
                label = lab),
            size = 2.8, colour = GREY,
            family = FONT, inherit.aes = FALSE) +
  scale_y_continuous(
    breaks = seq(0.50, 1.00, by = 0.05),
    labels = function(x) sprintf("%.2f", x)) +
  coord_cartesian(ylim = c(ylo, yhi)) +
  labs(x = NULL,
       y = "Weight on the EHR modality",
       title = paste("Fold-Selected Weight on the EHR",
                     "Modality, by Endpoint")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text.x = element_text(size = 11,
                               colour = DARK),
    axis.text.y = element_text(size = 9.5,
                               colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    plot.margin = margin(6, 10, 4, 4))

print(f2)
save_fig(f2, "fig19_twoway_weights", 5.8, 4.0)



























# ---------------------------------------------------------
# fig19_twoway_weights.R
# Fold-selected weight on the EHR modality, two-modality
# fusion
# ---------------------------------------------------------

library(ggplot2)
library(ggbeeswarm)
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

BLUE <- "#0072B2"
DARK <- "#333333"
GREY <- "#7F7F7F"

OUT_ORD <- c("composite_30d", "death_30d", "cv_first")

OUT_LAB <- c(composite_30d = "Composite outcome",
             death_30d = "30-day death",
             cv_first = "CV readmission")

GAP <- c("Composite outcome" = 0.0793,
         "30-day death"      = 0.1003,
         "CV readmission"    = 0.1073)

## ---- 2. data --------------------------------------------

wt <- read.csv(file.path(data_dir,
                         "twoway_weights.csv"),
               stringsAsFactors = FALSE)

cat("\n--- weights (fold 0 = oracle) ---\n")
print(wt, digits = 2)

keep_w <- intersect(OUT_ORD, unique(wt$outcome))

folds <- wt %>%
  filter(fold > 0, outcome %in% keep_w) %>%
  mutate(outcome = factor(outcome, levels = keep_w,
                          labels = OUT_LAB[keep_w]))

orc <- wt %>%
  filter(fold == 0, outcome %in% keep_w) %>%
  mutate(outcome = factor(outcome, levels = keep_w,
                          labels = OUT_LAB[keep_w]))

cat("\n--- weight against gap ---\n")
print(folds %>%
        group_by(outcome) %>%
        summarise(folds = n(),
                  modal = names(sort(table(w_ehr),
                                     decreasing = TRUE))[1],
                  min = min(w_ehr), max = max(w_ehr),
                  .groups = "drop") %>%
        mutate(gap = GAP[as.character(outcome)]) %>%
        as.data.frame(), digits = 3)

## Tight limits: nothing falls below 0.70, so start there.
ylo <- min(c(folds$w_ehr, orc$w_ehr)) - 0.022
yhi <- min(1.0, max(c(folds$w_ehr, orc$w_ehr)) + 0.055)

ANN <- data.frame(outcome = factor(names(GAP),
                                   levels = OUT_LAB[keep_w]),
                  lab = sprintf("gap %.3f", GAP)) %>%
  filter(!is.na(outcome)) %>%
  mutate(y = yhi - 0.012)

## ---- 3. plot --------------------------------------------

f2 <- ggplot(folds, aes(x = outcome, y = w_ehr)) +
  geom_errorbar(data = orc,
                aes(x = outcome, ymin = w_ehr,
                    ymax = w_ehr),
                width = 0.52, linewidth = 0.5,
                colour = DARK,
                inherit.aes = FALSE) +
  geom_beeswarm(size = 2.8, colour = BLUE,
                alpha = 0.9, cex = 2.6,
                method = "center") +
  geom_text(data = ANN,
            aes(x = outcome, y = y, label = lab),
            size = 2.8, colour = GREY,
            family = FONT, inherit.aes = FALSE) +
  scale_y_continuous(
    breaks = seq(0.50, 1.00, by = 0.05),
    labels = function(x) sprintf("%.2f", x)) +
  coord_cartesian(ylim = c(ylo, yhi)) +
  labs(x = NULL,
       y = "Weight on the EHR modality",
       title = paste("Fold-Selected Weight on the EHR",
                     "Modality, by Endpoint")) +
  theme(
    plot.title = element_text(size = 12,
                              face = "bold",
                              hjust = 0.5,
                              margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.y = element_text(size = 12,
                                colour = DARK,
                                margin = margin(r = 2)),
    axis.text.x = element_text(size = 11,
                               colour = DARK),
    axis.text.y = element_text(size = 9.5,
                               colour = DARK),
    axis.ticks.length = unit(0.06, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E8E8E8", linewidth = 0.3),
    plot.margin = margin(6, 10, 4, 4))

print(f2)
save_fig(f2, "fig19_twoway_weights", 5.8, 4.0)





























# ---------------------------------------------------------
# fig20_weights2.R
# Division of the fusion weight between the two modalities
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
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

MOD_LAB <- c(w_ehr = "EHR", w_ecg = "ECG")

## Okabe-Ito, matching the modality colours used throughout
PAL <- c("EHR" = "#0072B2",
         "ECG" = "#E69F00")

## darker variants, for text on a white plate
PAL_TXT <- c("EHR" = "#00446B",
             "ECG" = "#8A5F00")

## ---- 2. weights -----------------------------------------

w <- read.csv(file.path(data_dir,
                        "twoway_weights.csv"),
              stringsAsFactors = FALSE)

## fold 0 is the global oracle weight; drop it
w <- w %>% filter(fold > 0)
w$w_ecg <- 1 - w$w_ehr

cat("\n--- fold weights ---\n")
print(w, digits = 2)

keep <- intersect(names(OUT_LAB), unique(w$outcome))
ORD <- rev(keep)

mw <- w %>%
  filter(outcome %in% keep) %>%
  group_by(outcome) %>%
  summarise(across(c(w_ehr, w_ecg), mean),
            .groups = "drop")

stopifnot(all(abs(mw$w_ehr + mw$w_ecg - 1) < 1e-8))

rng <- w %>%
  filter(outcome %in% keep) %>%
  group_by(outcome) %>%
  summarise(lo = min(w_ehr), hi = max(w_ehr),
            .groups = "drop") %>%
  mutate(lab = ifelse(
    lo == hi,
    sprintf("EHR %.2f in every fold", lo),
    sprintf("EHR %.2f\u2013%.2f across folds", lo, hi)),
    yy = match(as.character(outcome), ORD))

cat("\n--- EHR weight range ---\n")
print(as.data.frame(rng))

d <- mw %>%
  pivot_longer(-outcome, names_to = "modality",
               values_to = "wt") %>%
  mutate(modality = factor(MOD_LAB[modality],
                           levels = unname(MOD_LAB))) %>%
  arrange(match(outcome, ORD), modality) %>%
  group_by(outcome) %>%
  mutate(hi = cumsum(wt), lo = hi - wt,
         mid = (lo + hi) / 2) %>%
  ungroup() %>%
  mutate(yy = match(as.character(outcome), ORD))

HH <- 0.30   # half-height of each bar

## ---- 3. pattern overlay for greyscale -------------------
## EHR solid; ECG vertical rules.

vlines <- do.call(rbind, lapply(
  which(d$modality == "ECG"), function(i) {
    r <- d[i, ]
    xs <- seq(r$lo + 0.008, r$hi - 0.008, by = 0.011)
    if (length(xs) < 1) return(NULL)
    data.frame(x = xs, yy = r$yy)
  }))

cat("\npattern overlays: ",
    ifelse(is.null(vlines), 0, nrow(vlines)),
    " rules\n", sep = "")

lab_d <- filter(d, wt >= 0.08)
top <- d %>% filter(yy == max(yy))

## ---- 4. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.2, 0.8, by = 0.2),
             colour = "grey90", linewidth = 0.3) +
  geom_rect(data = d,
            aes(xmin = lo, xmax = hi, fill = modality,
                ymin = yy - HH, ymax = yy + HH),
            colour = "white", linewidth = 0.7)

if (!is.null(vlines)) {
  f <- f + geom_segment(
    data = vlines,
    aes(x = x, xend = x,
        y = yy - HH + 0.02, yend = yy + HH - 0.02),
    colour = "white", linewidth = 0.35,
    inherit.aes = FALSE)
}

f <- f +
  geom_label(data = lab_d,
             aes(x = mid, y = yy,
                 label = sprintf("%.2f", wt),
                 colour = modality),
             fill = "white", label.size = 0,
             label.r = unit(0.10, "lines"),
             label.padding = unit(0.16, "lines"),
             fontface = "bold", size = 3.4,
             family = FONT, show.legend = FALSE) +
  geom_text(data = top,
            aes(x = mid, y = yy + HH + 0.24,
                label = modality, colour = modality),
            fontface = "bold", size = 3.2,
            family = FONT, show.legend = FALSE) +
  geom_text(data = rng,
            aes(x = 1.02, y = yy, label = lab),
            hjust = 0, size = 3.1, colour = "grey30",
            family = FONT) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL_TXT,
                      guide = "none") +
  scale_x_continuous(
    limits = c(0, 1.42),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(ORD),
    labels = OUT_LAB[ORD],
    limits = c(0.5, length(ORD) + 0.95),
    expand = c(0, 0)) +
  labs(x = "Share of the fusion weight (modalities sum to 1.0)",
       y = NULL,
       title = paste("Division of the Fusion Weight Between",
                     "the EHR and ECG Modalities")) +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 10,
                                   colour = "#333333"),
        axis.text.x = element_text(size = 9.5,
                                   colour = "#7F7F7F"),
        axis.title.x = element_text(
          size = 12, colour = "#333333",
          margin = margin(t = 4)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

print(f)
save_fig(f, "fig20_weights2", 8.4, 3.4)































# ---------------------------------------------------------
# fig20_weights2.R
# Division of the fusion weight between the two modalities
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
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

OUT_LAB <- c(composite_30d    = "Composite (30-Day)",
             death_30d        = "30-Day Death",
             death_30d_inhosp = "In-Hospital Death",
             death_inhosp     = "In-Hospital Death",
             composite_inhosp = "In-Hospital Composite",
             cv_first         = "CV Readmission",
             cv_30d           = "CV Readmission")

## display order, top to bottom
DISPLAY <- c("Composite (30-Day)", "30-Day Death",
             "In-Hospital Death", "CV Readmission")

MOD_LAB <- c(w_ehr = "EHR", w_ecg = "ECG")

## Okabe-Ito, matching the modality colours used throughout
PAL <- c("EHR" = "#0072B2",
         "ECG" = "#E69F00")

## darker variants, for text on a white plate
PAL_TXT <- c("EHR" = "#00446B",
             "ECG" = "#8A5F00")

## ---- 2. weights -----------------------------------------

w <- read.csv(file.path(data_dir,
                        "twoway_weights.csv"),
              stringsAsFactors = FALSE)

cat("\noutcome keys in file: ",
    paste(unique(w$outcome), collapse = " | "), "\n",
    sep = "")

miss <- setdiff(unique(w$outcome), names(OUT_LAB))
if (length(miss) > 0) {
  message("NO LABEL for: ", paste(miss, collapse = ", "),
          " - add these keys to OUT_LAB")
}

## fold 0 is the global oracle weight; drop it
w <- w %>% filter(fold > 0)
w$w_ecg <- 1 - w$w_ehr

w$label <- OUT_LAB[w$outcome]
w <- w %>% filter(!is.na(label))

ORD <- rev(intersect(DISPLAY, unique(w$label)))
cat("outcomes plotted: ",
    paste(rev(ORD), collapse = " | "), "\n\n", sep = "")

cat("--- fold weights ---\n")
print(w[, c("label", "fold", "w_ehr", "w_ecg")],
      digits = 2, row.names = FALSE)

mw <- w %>%
  group_by(label) %>%
  summarise(across(c(w_ehr, w_ecg), mean),
            .groups = "drop")

stopifnot(all(abs(mw$w_ehr + mw$w_ecg - 1) < 1e-8))

rng <- w %>%
  group_by(label) %>%
  summarise(lo = min(w_ehr), hi = max(w_ehr),
            .groups = "drop") %>%
  mutate(txt = ifelse(
    lo == hi,
    sprintf("EHR %.2f in every fold", lo),
    sprintf("EHR %.2f\u2013%.2f across folds", lo, hi)),
    yy = match(as.character(label), ORD))

cat("\n--- EHR weight range ---\n")
print(as.data.frame(rng), digits = 3)

d <- mw %>%
  pivot_longer(-label, names_to = "modality",
               values_to = "wt") %>%
  mutate(modality = factor(MOD_LAB[modality],
                           levels = unname(MOD_LAB))) %>%
  arrange(match(label, ORD), modality) %>%
  group_by(label) %>%
  mutate(hi = cumsum(wt), lo = hi - wt,
         mid = (lo + hi) / 2) %>%
  ungroup() %>%
  mutate(yy = match(as.character(label), ORD))

HH <- 0.30   # half-height of each bar

## ---- 3. pattern overlay for greyscale -------------------
## EHR solid; ECG vertical rules.

vlines <- do.call(rbind, lapply(
  which(d$modality == "ECG"), function(i) {
    r <- d[i, ]
    xs <- seq(r$lo + 0.008, r$hi - 0.008, by = 0.011)
    if (length(xs) < 1) return(NULL)
    data.frame(x = xs, yy = r$yy)
  }))

cat("\npattern overlays: ",
    ifelse(is.null(vlines), 0, nrow(vlines)),
    " rules\n", sep = "")

lab_d <- filter(d, wt >= 0.08)
top <- d %>% filter(yy == max(yy))

## ---- 4. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.2, 0.8, by = 0.2),
             colour = "grey90", linewidth = 0.3) +
  geom_rect(data = d,
            aes(xmin = lo, xmax = hi, fill = modality,
                ymin = yy - HH, ymax = yy + HH),
            colour = "white", linewidth = 0.7)

if (!is.null(vlines)) {
  f <- f + geom_segment(
    data = vlines,
    aes(x = x, xend = x,
        y = yy - HH + 0.02, yend = yy + HH - 0.02),
    colour = "white", linewidth = 0.35,
    inherit.aes = FALSE)
}

f <- f +
  geom_label(data = lab_d,
             aes(x = mid, y = yy,
                 label = sprintf("%.2f", wt),
                 colour = modality),
             fill = "white", label.size = 0,
             label.r = unit(0.10, "lines"),
             label.padding = unit(0.16, "lines"),
             fontface = "bold", size = 3.4,
             family = FONT, show.legend = FALSE) +
  geom_text(data = top,
            aes(x = mid, y = yy + HH + 0.24,
                label = modality, colour = modality),
            fontface = "bold", size = 3.2,
            family = FONT, show.legend = FALSE) +
  geom_text(data = rng,
            aes(x = 1.02, y = yy, label = txt),
            hjust = 0, size = 3.1, colour = "grey30",
            family = FONT) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL_TXT,
                      guide = "none") +
  scale_x_continuous(
    limits = c(0, 1.42),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(ORD),
    labels = ORD,
    limits = c(0.5, length(ORD) + 0.95),
    expand = c(0, 0)) +
  labs(x = "Share of the fusion weight (modalities sum to 1.0)",
       y = NULL,
       title = paste("Division of the Fusion Weight Between",
                     "the EHR and ECG Modalities")) +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 10,
                                   colour = "#333333"),
        axis.text.x = element_text(size = 9.5,
                                   colour = "#7F7F7F"),
        axis.title.x = element_text(
          size = 12, colour = "#333333",
          margin = margin(t = 4)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

print(f)
save_fig(f, "fig20_weights2", 8.4,
         0.62 * length(ORD) + 1.4)

















# ---------------------------------------------------------
# fig20_weights2.R
# Division of the fusion weight between the two modalities
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(grid)
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

theme_set(theme_minimal(base_size = 11,
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

## Okabe-Ito, matching the modality colours used throughout
PAL <- c("EHR" = "#0072B2",
         "ECG" = "#E69F00")

## darker variants, for text on a white plate
PAL_TXT <- c("EHR" = "#00446B",
             "ECG" = "#8A5F00")

DARK <- "#333333"
GREY <- "#7F7F7F"

## ---- 2. fold weights on the EHR modality ----------------

folds <- tribble(
  ~outcome,             ~w,
  "Composite (30-Day)", c(0.70, 0.75, 0.75, 0.70, 0.75),
  "30-Day Death",       c(0.75, 0.75, 0.75, 0.70, 0.70),
  "In-Hospital Death",  c(0.80, 0.75, 0.75, 0.80, 0.80),
  "CV Readmission",     c(0.80, 0.85, 0.85, 0.85, 0.85)
)

ORD <- rev(folds$outcome)

mw <- folds %>%
  mutate(w_ehr = sapply(w, mean),
         w_ecg = 1 - w_ehr,
         lo = sapply(w, min),
         hi = sapply(w, max)) %>%
  select(-w)

cat("\n--- mean weights ---\n")
print(as.data.frame(mw), digits = 3)
cat("outcomes plotted:", nrow(mw), "\n")

rng <- mw %>%
  transmute(outcome,
            lab = sprintf("EHR %.2f\u2013%.2f across folds",
                          lo, hi),
            yy = match(outcome, ORD))

d <- mw %>%
  select(outcome, w_ehr, w_ecg) %>%
  pivot_longer(-outcome, names_to = "modality",
               values_to = "wt") %>%
  mutate(modality = factor(
    ifelse(modality == "w_ehr", "EHR", "ECG"),
    levels = c("EHR", "ECG"))) %>%
  arrange(match(outcome, ORD), modality) %>%
  group_by(outcome) %>%
  mutate(hi = cumsum(wt), lo = hi - wt,
         mid = (lo + hi) / 2) %>%
  ungroup() %>%
  mutate(yy = match(outcome, ORD))

HH <- 0.30

## ---- 3. pattern overlay for greyscale -------------------
## EHR solid; ECG vertical rules.

vlines <- do.call(rbind, lapply(
  which(d$modality == "ECG"), function(i) {
    r <- d[i, ]
    xs <- seq(r$lo + 0.008, r$hi - 0.008, by = 0.011)
    if (length(xs) < 1) return(NULL)
    data.frame(x = xs, yy = r$yy)
  }))

top <- d %>% filter(yy == max(yy))

## ---- 4. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.2, 0.8, by = 0.2),
             colour = "grey90", linewidth = 0.3) +
  geom_rect(data = d,
            aes(xmin = lo, xmax = hi, fill = modality,
                ymin = yy - HH, ymax = yy + HH),
            colour = "white", linewidth = 0.7)

if (!is.null(vlines)) {
  f <- f + geom_segment(
    data = vlines,
    aes(x = x, xend = x,
        y = yy - HH + 0.02, yend = yy + HH - 0.02),
    colour = "white", linewidth = 0.35,
    inherit.aes = FALSE)
}

f <- f +
  geom_label(data = filter(d, wt >= 0.08),
             aes(x = mid, y = yy,
                 label = sprintf("%.2f", wt),
                 colour = modality),
             fill = "white", label.size = 0,
             label.r = unit(0.10, "lines"),
             label.padding = unit(0.16, "lines"),
             fontface = "bold", size = 3.4,
             family = FONT, show.legend = FALSE) +
  geom_text(data = top,
            aes(x = mid, y = yy + HH + 0.24,
                label = modality, colour = modality),
            fontface = "bold", size = 3.2,
            family = FONT, show.legend = FALSE) +
  geom_text(data = rng,
            aes(x = 1.02, y = yy, label = lab),
            hjust = 0, size = 3.1, colour = "grey30",
            family = FONT) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL_TXT,
                      guide = "none") +
  scale_x_continuous(
    limits = c(0, 1.44),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(ORD), labels = ORD,
    limits = c(0.5, length(ORD) + 0.95),
    expand = c(0, 0)) +
  labs(x = "Share of the fusion weight (modalities sum to 1.0)",
       y = NULL,
       title = paste("Division of the Fusion Weight Between",
                     "the EHR and ECG Modalities")) +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 10,
                                   colour = DARK),
        axis.text.x = element_text(size = 9.5,
                                   colour = GREY),
        axis.title.x = element_text(
          size = 12, colour = DARK,
          margin = margin(t = 4)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

print(f)
save_fig(f, "fig20_weights2", 8.6, 4.0)



































# ---------------------------------------------------------
# fig21_w2_pairings.R
# Weight split for the two candidate second modalities,
# on the subset where both are available
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(grid)
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

theme_set(theme_minimal(base_size = 11,
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

## Okabe-Ito, matching the modality colours used throughout
PAL <- c("EHR"             = "#0072B2",
         "Second modality" = "#009E73")
PAL_TXT <- c("EHR"             = "#00446B",
             "Second modality" = "#00604A")

DARK <- "#333333"
GREY <- "#7F7F7F"

## ---- 2. fold weights on the EHR modality ----------------

folds <- tribble(
  ~outcome,             ~pair,   ~w,
  "Composite (30-Day)", "+ CTPA report",
  c(0.65, 0.70, 0.75, 0.70, 0.70),
  "Composite (30-Day)", "+ ECG",
  c(0.75, 0.80, 0.75, 0.75, 0.75),
  "30-Day Death", "+ CTPA report",
  c(0.70, 0.75, 0.65, 0.70, 0.70),
  "30-Day Death", "+ ECG",
  c(0.80, 0.75, 0.75, 0.75, 0.75),
  "CV Readmission", "+ CTPA report",
  c(0.95, 1.00, 0.95, 1.00, 0.95),
  "CV Readmission", "+ ECG",
  c(1.00, 1.00, 0.90, 1.00, 0.80)
)

OUT_ORD <- c("Composite (30-Day)", "30-Day Death",
             "CV Readmission")

mw <- folds %>%
  mutate(w_ehr = sapply(w, mean),
         w_2nd = 1 - w_ehr,
         lo2 = 1 - sapply(w, max),
         hi2 = 1 - sapply(w, min)) %>%
  select(-w)

cat("\n--- mean weight on the second modality ---\n")
print(as.data.frame(
  mw %>% select(outcome, pair, w_2nd, lo2, hi2)),
  digits = 3)

lvl <- rev(paste(rep(OUT_ORD, each = 2),
                 c("+ CTPA report", "+ ECG"),
                 sep = " | "))

mw <- mw %>%
  mutate(key = paste(outcome, pair, sep = " | "),
         key = factor(key, levels = lvl),
         yy = as.numeric(key))

rng <- mw %>%
  transmute(yy,
            lab = sprintf(
              "second modality %.2f\u2013%.2f", lo2, hi2))

grp <- mw %>%
  group_by(outcome) %>%
  summarise(yy = mean(yy), .groups = "drop") %>%
  mutate(outcome = factor(outcome, levels = OUT_ORD))

## Faint separators between endpoint groups.
seps <- mw %>%
  group_by(outcome) %>%
  summarise(top = max(yy), .groups = "drop") %>%
  mutate(y = top + 0.5) %>%
  filter(y < max(mw$yy) + 0.4)

d <- mw %>%
  select(key, yy, w_ehr, w_2nd) %>%
  pivot_longer(c(w_ehr, w_2nd), names_to = "modality",
               values_to = "wt") %>%
  mutate(modality = factor(
    ifelse(modality == "w_ehr", "EHR",
           "Second modality"),
    levels = c("EHR", "Second modality"))) %>%
  arrange(yy, modality) %>%
  group_by(yy) %>%
  mutate(hi = cumsum(wt), lo = hi - wt,
         mid = (lo + hi) / 2) %>%
  ungroup()

HH <- 0.32

## ---- 3. pattern overlay for greyscale -------------------
## EHR solid; second modality stippled.

dots <- do.call(rbind, lapply(
  which(d$modality == "Second modality"), function(i) {
    r <- d[i, ]
    xs <- seq(r$lo + 0.010, r$hi - 0.010, by = 0.016)
    if (length(xs) < 1) return(NULL)
    expand.grid(x = xs, dy = c(-0.16, 0, 0.16)) %>%
      mutate(yy = r$yy + dy)
  }))

top <- d %>% filter(yy == max(yy))

lab_y <- mw %>%
  transmute(yy, lab = sub(".*\\| ", "", key))

## ---- 4. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.2, 0.8, by = 0.2),
             colour = "grey90", linewidth = 0.3) +
  geom_hline(data = seps, aes(yintercept = y),
             colour = "grey88", linewidth = 0.4,
             inherit.aes = FALSE) +
  geom_rect(data = d,
            aes(xmin = lo, xmax = hi, fill = modality,
                ymin = yy - HH, ymax = yy + HH),
            colour = "white", linewidth = 0.7)

if (!is.null(dots)) {
  f <- f + geom_point(
    data = dots, aes(x = x, y = yy),
    colour = "white", size = 0.55,
    inherit.aes = FALSE)
}

f <- f +
  geom_label(data = filter(d, wt >= 0.08),
             aes(x = mid, y = yy,
                 label = sprintf("%.2f", wt),
                 colour = modality),
             fill = "white", label.size = 0,
             label.r = unit(0.10, "lines"),
             label.padding = unit(0.16, "lines"),
             fontface = "bold", size = 3.3,
             family = FONT, show.legend = FALSE) +
  geom_text(data = top,
            aes(x = mid, y = yy + HH + 0.26,
                label = modality, colour = modality),
            fontface = "bold", size = 3.2,
            family = FONT, show.legend = FALSE) +
  geom_text(data = rng,
            aes(x = 1.02, y = yy, label = lab),
            hjust = 0, size = 3.0, colour = "grey30",
            family = FONT) +
  geom_text(data = grp,
            aes(x = -0.30, y = yy, label = outcome),
            hjust = 0.5, size = 3.2, fontface = "bold",
            colour = DARK, family = FONT,
            inherit.aes = FALSE) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL_TXT,
                      guide = "none") +
  scale_x_continuous(
    limits = c(-0.60, 1.50),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = lab_y$yy, labels = lab_y$lab,
    limits = c(0.4, max(d$yy) + 0.95),
    expand = c(0, 0)) +
  labs(x = "Share of the fusion weight (modalities sum to 1.0)",
       y = NULL,
       title = paste("Division of the Fusion Weight Between",
                     "the EHR and Each Candidate Second",
                     "Modality")) +
  coord_cartesian(clip = "off") +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 9.5,
                                   colour = DARK),
        axis.text.x = element_text(size = 9.5,
                                   colour = GREY),
        axis.title.x = element_text(
          size = 12, colour = DARK,
          margin = margin(t = 4)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

print(f)
save_fig(f, "fig21_w2_pairings", 9.2, 4.6)





























# ---------------------------------------------------------
# fig22_twoway_disagree.R
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

POS  <- "#0072B2"   # Okabe-Ito blue
NEG  <- "#D55E00"   # Okabe-Ito vermillion
DARK <- "#333333"

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
  labs(x = "Inter-modality disagreement tercile",
       y = "Gain in AUROC over the better single modality",
       title = paste("Fusion Gain by Inter-Modality",
                     "Disagreement Tercile")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        strip.text = element_text(size = 10.5,
                                  face = "bold",
                                  colour = DARK,
                                  margin = margin(b = 4)),
        axis.text = element_text(size = 9.5,
                                 colour = DARK),
        axis.title.x = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(t = 2)),
        axis.title.y = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(r = 2)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 22...\n")
print(system.time(
  save_fig(f, "fig22_twoway_disagree", 9.2, 3.9)))

























# ---------------------------------------------------------
# fig30_time_to_event.R
# Distribution of days from admission to first event
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

BLUE <- "#0072B2"
DARK <- "#333333"
GREY <- "#7F7F7F"

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "decomp_p2_events.csv"),
              stringsAsFactors = FALSE)

cat("\nevents:", nrow(d), "\n")
cat(sprintf("mean %.2f  median %.1f  IQR %.0f-%.0f\n",
            mean(d$days), median(d$days),
            quantile(d$days, .25),
            quantile(d$days, .75)))
print(table(d$route))

md <- median(d$days)
mu <- mean(d$days)
lab <- sprintf("%d events, median %.0f days",
               nrow(d), md)

## headroom above the tallest bar for the annotations
top <- max(table(cut(d$days, seq(-0.5, 30.5, 1))))
YMAX <- top * 1.30

f <- ggplot(d, aes(x = days)) +
  geom_vline(xintercept = c(7.5, 14.5),
             colour = "grey70", linewidth = 0.4,
             linetype = "22") +
  geom_histogram(binwidth = 1, boundary = 0,
                 fill = BLUE, colour = "white",
                 linewidth = 0.25) +
  geom_vline(xintercept = md, colour = BLUE,
             linewidth = 0.6) +
  ## summary top-right, matching Figure 31
  annotate("text", x = 30, y = YMAX * 0.965,
           label = lab, hjust = 1, vjust = 1,
           size = 3.0, colour = BLUE,
           family = FONT) +
  ## window labels on their own row, below the summary
  annotate("text", x = c(3.75, 11, 22.5),
           y = YMAX * 0.855, vjust = 1, size = 2.8,
           colour = "grey45", family = FONT,
           label = c("0-7 days", "8-14 days",
                     "15-30 days")) +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(0, YMAX),
    breaks = scales::pretty_breaks(5),
    expand = c(0, 0)) +
  labs(x = "Days from admission to first event",
       y = "Number of admissions",
       title = paste("Distribution of Days From Admission",
                     "to First Composite Event")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9.5,
                                 colour = DARK),
        axis.title.x = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(t = 2)),
        axis.title.y = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(r = 2)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 30...\n")
print(system.time(
  save_fig(f, "fig30_time_to_event", 7.6, 4.0)))





























# ---------------------------------------------------------
# fig31_time_to_event_route.R
# Distribution of days from admission to first event,
# by ascertainment route
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

PAL <- c("All-Cause Death" = "#0072B2",
         "CV Readmission"  = "#D55E00")

DARK <- "#333333"

## ---- 2. data --------------------------------------------

d <- read.csv(file.path(data_dir,
                        "decomp_p2_events.csv"),
              stringsAsFactors = FALSE)

pd <- d %>%
  mutate(route = ifelse(route == "Death",
                        "All-Cause Death",
                        "CV Readmission"),
         route = factor(route, levels = names(PAL)))

cat("\n--- by route ---\n")
print(pd %>% group_by(route) %>%
        summarise(n = n(), mean = mean(days),
                  median = median(days),
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

cat("\nroute by window:\n")
print(table(pd$route,
            cut(pd$days, c(-1, 7, 14, 30),
                labels = c("0-7d", "8-14d",
                           "15-30d"))))

lab <- pd %>%
  group_by(route) %>%
  summarise(n = n(), md = median(days),
            .groups = "drop") %>%
  mutate(txt = sprintf("%d events, median %.0f days",
                       n, md))

f <- ggplot(pd, aes(x = days, fill = route)) +
  geom_vline(xintercept = c(7.5, 14.5),
             colour = "grey70", linewidth = 0.4,
             linetype = "22") +
  geom_histogram(binwidth = 1, boundary = 0,
                 colour = "white", linewidth = 0.25) +
  geom_vline(data = lab,
             aes(xintercept = md, colour = route),
             linewidth = 0.6, show.legend = FALSE) +
  geom_text(data = lab,
            aes(x = 30, y = Inf, label = txt,
                colour = route),
            hjust = 1, vjust = 1.9, size = 3.0,
            family = FONT, show.legend = FALSE) +
  facet_wrap(~ route, ncol = 1) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_x_continuous(
    limits = c(-0.5, 30.5),
    breaks = seq(0, 30, by = 5),
    expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(
    mult = c(0, 0.20))) +
  labs(x = "Days from admission to first event",
       y = "Number of admissions",
       title = paste("Distribution of Days From Admission",
                     "to First Event, by Ascertainment",
                     "Route")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.y = unit(0.9, "lines"),
        strip.text = element_text(size = 10.5,
                                  face = "bold",
                                  colour = DARK,
                                  margin = margin(b = 4)),
        axis.text = element_text(size = 9.5,
                                 colour = DARK),
        axis.title.x = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(t = 2)),
        axis.title.y = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(r = 2)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 31...\n")
print(system.time(
  save_fig(f, "fig31_time_to_event_route", 7.6, 5.4)))




















# ---------------------------------------------------------
# fig32_fusion_architectures.R
# Combination rules against the weighted average,
# three modalities
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

NS   <- "#0072B2"   # Okabe-Ito blue: interval spans zero
SIG  <- "#D55E00"   # Okabe-Ito vermillion: excludes zero
DARK <- "#333333"

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
  labs(x = paste("Change in AUROC relative to the",
                 "weighted average"),
       y = NULL,
       title = paste("Combination Rules Against the",
                     "Weighted Average, Three",
                     "Modalities")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey95", linewidth = 0.3,
          linetype = "dotted"),
        panel.grid.major.x = element_line(
          colour = "grey93", linewidth = 0.3),
        panel.spacing.y = unit(0.35, "lines"),
        panel.spacing.x = unit(0.7, "lines"),
        strip.text.x = element_text(size = 10.5,
                                    face = "bold",
                                    colour = DARK,
                                    margin = margin(b = 4)),
        strip.text.y = element_text(
          size = 9, face = "bold", angle = 0,
          hjust = 0, colour = "grey30"),
        axis.text.y = element_text(size = 9.5,
                                   colour = DARK),
        axis.text.x = element_text(size = 9.5,
                                   colour = DARK),
        axis.title.x = element_text(
          size = 12, colour = DARK,
          margin = margin(t = 4)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 32...\n")
print(system.time(
  save_fig(f, "fig32_fusion_architectures", 9.4, 4.4)))





















# ---------------------------------------------------------
# fig33_transfer_cost.R
# Feature-restriction cost against institution cost
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

PAL <- c("Restricting the feature set"     = "#9ECAE1",
         "Training at another institution" = "#08519C")
TXT <- c("Restricting the feature set"     = "#2171B5",
         "Training at another institution" = "#08306B")

DARK <- "#333333"

N_RULE <- 4     # fixed rules per institution segment

## ---- 2. data --------------------------------------------

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
    xs <- seq(r$lo, r$hi,
              length.out = N_RULE + 2)
    xs <- xs[c(-1, -length(xs))]
    data.frame(x = xs, yy = r$yy)
  }))

lab_top <- pd %>% filter(yy == max(yy))

## ---- 3. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.02, 0.12, by = 0.02),
             colour = "grey93", linewidth = 0.3) +
  geom_rect(data = pd,
            aes(xmin = lo, xmax = hi, fill = component,
                ymin = yy - HH, ymax = yy + HH),
            colour = "grey25", linewidth = 0.3) +
  geom_segment(data = hatch,
               aes(x = x, xend = x,
                   y = yy - HH + 0.015,
                   yend = yy + HH - 0.015),
               colour = "white", linewidth = 0.35,
               inherit.aes = FALSE) +
  geom_text(data = filter(pd,
                          component == names(PAL)[1]),
            aes(x = mid, y = yy,
                label = sprintf("%.4f", cost)),
            colour = "grey15", fontface = "bold",
            size = 3.0, family = FONT) +
  geom_text(data = filter(pd,
                          component == names(PAL)[2]),
            aes(x = mid, y = yy,
                label = sprintf("%.4f", cost)),
            colour = "white", fontface = "bold",
            size = 3.0, family = FONT) +
  geom_text(data = pd %>% distinct(outcome, total,
                                   ratio, yy),
            aes(x = total + 0.003, y = yy,
                label = sprintf("%.1f\u00d7", ratio)),
            hjust = 0, fontface = "bold", size = 3.1,
            colour = "grey25", family = FONT) +
  geom_text(data = lab_top,
            aes(x = mid, y = yy + HH + 0.23,
                label = component, colour = component),
            fontface = "bold", size = 3.1,
            family = FONT, show.legend = FALSE) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = TXT, guide = "none") +
  scale_x_continuous(
    limits = c(0, 0.142),
    breaks = seq(0, 0.12, by = 0.02),
    labels = function(x) sprintf("%.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(OUT_ORD), labels = OUT_ORD,
    limits = c(0.5, length(OUT_ORD) + 0.85),
    expand = c(0, 0)) +
  labs(x = "Cost in AUROC",
       y = NULL,
       title = paste("Decomposition of Transfer Cost Into",
                     "Feature-Restriction and Institution",
                     "Components")) +
  theme(panel.grid = element_blank(),
        axis.text = element_text(size = 9.5,
                                 colour = DARK),
        axis.title.x = element_text(
          size = 12, colour = DARK,
          margin = margin(t = 4)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 33...\n")
print(system.time(
  save_fig(f, "fig33_transfer_cost", 7.8, 3.4)))





























# ---------------------------------------------------------
# fig34_corr_shift.R
# Feature correlation at the source against the target
# institution
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
SHIFT  <- "#D55E00"   # Okabe-Ito vermillion
DARK   <- "#333333"

N_LAB <- 8            # divergences to name
REV_MIN <- 0.03       # magnitude floor for a reversal

## data region ends at 1.02; margin runs beyond it
X_MAX <- 1.02
X_TICK <- 1.10        # leader lines converge here
X_TEXT <- 1.14        # label column starts here
X_LIM <- 2.05         # full canvas including margin

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

lab <- pd %>%
  filter(big) %>%
  arrange(desc(target)) %>%
  mutate(ty = seq(0.60, -0.42,
                  length.out = n()),
         txt = sprintf("%s  (%+.2f to %+.2f)",
                       pair, source, target))

print(lab %>% select(pair, source, target, diff,
                     reverse) %>%
        as.data.frame(), digits = 3)

Y_LIM <- c(-0.62, 1.02)

## ---- 3. plot and save -----------------------------------

f <- ggplot() +
  ## data region
  geom_abline(slope = 1, intercept = 0,
              colour = "grey55", linewidth = 0.45,
              linetype = "22") +
  geom_hline(yintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey88",
             linewidth = 0.3) +
  geom_point(data = filter(pd, !reverse),
             aes(x = source, y = target),
             shape = 16, size = 1.7, alpha = 0.45,
             colour = STABLE) +
  geom_point(data = filter(pd, reverse),
             aes(x = source, y = target),
             shape = 22, size = 2.5, stroke = 0.9,
             colour = SHIFT, fill = "white") +
  ## boundary between data region and label margin
  geom_vline(xintercept = X_MAX, colour = "grey80",
             linewidth = 0.4) +
  ## leader lines: point -> converge -> label
  geom_segment(data = lab,
               aes(x = source, y = target,
                   xend = X_TICK, yend = ty),
               colour = "grey65", linewidth = 0.28) +
  geom_point(data = lab,
             aes(x = X_TICK, y = ty),
             shape = 22, size = 1.8, stroke = 0.7,
             colour = SHIFT, fill = "white") +
  geom_text(data = lab,
            aes(x = X_TEXT, y = ty, label = txt),
            hjust = 0, size = 2.85, colour = SHIFT,
            family = FONT) +
  annotate("text", x = X_TEXT, y = 0.76, hjust = 0,
           label = "Eight largest divergences",
           size = 3.0, fontface = "bold",
           colour = "grey20", family = FONT) +
  annotate("text", x = X_TEXT, y = 0.68, hjust = 0,
           label = "source to target correlation",
           size = 2.7, colour = "grey45",
           family = FONT) +
  annotate("text", x = -0.58, y = 0.97, hjust = 0,
           label = paste("hollow squares reverse sign;",
                         "points on the line are",
                         "unchanged"),
           size = 2.85, colour = "grey35",
           family = FONT) +
  scale_x_continuous(
    limits = c(-0.62, X_LIM),
    breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    limits = Y_LIM,
    breaks = seq(-0.5, 1.0, by = 0.25),
    labels = function(x) sprintf("%+.2f", x),
    expand = c(0, 0)) +
  labs(x = "Correlation at the source institution",
       y = "Correlation at the target institution",
       title = paste("Feature Pair Correlations at the",
                     "Source Against the Target",
                     "Institution")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey94", linewidth = 0.3),
        axis.text = element_text(size = 9.5,
                                 colour = DARK),
        axis.title.x = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(t = 2)),
        axis.title.y = element_text(size = 12,
                                    colour = DARK,
                                    margin = margin(r = 2)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.28,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 34...\n")
print(system.time(
  save_fig(f, "fig34_corr_shift", 10.4, 6.4)))






























# ---------------------------------------------------------
# fig35_supervision_ladder.R
# Fusion increment as the EHR modality is strengthened
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

PAL <- c("30-Day Death" = "#0072B2",
         "Composite (30-Day)" = "#D55E00")
SHP <- c("30-Day Death" = 16,
         "Composite (30-Day)" = 17)
LTY <- c("30-Day Death" = "solid",
         "Composite (30-Day)" = "22")

DARK <- "#333333"

RUNG <- c("Transferred\n28 features",
          "In-domain\n28 features",
          "In-domain\nall features")

NUDGE <- 0.20   # horizontal offset for value labels

## ---- 2. data --------------------------------------------

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
  mutate(outcome = factor(outcome,
                          levels = names(PAL)),
         rungn = rung,
         rung = factor(rung, levels = 1:3,
                       labels = RUNG),
         lx = ifelse(rungn == 3,
                     rungn - NUDGE, rungn + NUDGE),
         hj = ifelse(rungn == 3, 1, 0))

cat("\n--- fusion increment by rung ---\n")
print(as.data.frame(pd %>%
  select(outcome, rung, inc, lo, hi)), digits = 3)
cat("\nratio between successive rungs:\n")
print(pd %>% group_by(outcome) %>%
        summarise(r1 = inc[1] / inc[2],
                  r2 = inc[2] / inc[3],
                  .groups = "drop") %>%
        as.data.frame(), digits = 3)

## ---- 3. plot and save -----------------------------------

f <- ggplot(pd, aes(x = rungn, y = inc,
                    colour = outcome,
                    shape = outcome,
                    linetype = outcome,
                    group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey40",
             linewidth = 0.5) +
  geom_line(linewidth = 0.75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.10,
                linewidth = 0.45, linetype = "solid",
                na.rm = TRUE) +
  geom_point(size = 3.0) +
  geom_text(aes(x = lx, y = inc, hjust = hj,
                label = sprintf("%+.4f", inc)),
            vjust = 0.5, size = 3.0,
            fontface = "bold", family = FONT,
            show.legend = FALSE) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY,
                        guide = "none") +
  scale_x_continuous(
    breaks = 1:3, labels = RUNG,
    limits = c(0.62, 3.38), expand = c(0, 0)) +
  scale_y_continuous(
    limits = c(-0.002, 0.046),
    breaks = seq(0, 0.04, by = 0.01),
    labels = function(x) sprintf("%+.3f", x),
    expand = c(0, 0)) +
  labs(x = "Configuration of the EHR modality",
       y = "Fusion increment over the EHR modality",
       title = paste("Fusion Increment by Configuration",
                     "of the EHR Modality")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(
          colour = "grey94", linewidth = 0.3),
        panel.spacing.x = unit(1.0, "lines"),
        strip.text = element_text(size = 10.5,
                                  face = "bold",
                                  colour = DARK,
                                  margin = margin(b = 4)),
        axis.text.x = element_text(size = 9,
                                   colour = DARK,
                                   lineheight = 0.95),
        axis.text.y = element_text(size = 9.5,
                                   colour = DARK),
        axis.title.x = element_text(
          size = 12, colour = DARK,
          margin = margin(t = 8)),
        axis.title.y = element_text(
          size = 12, colour = DARK,
          margin = margin(r = 2)),
        axis.ticks.length = unit(0.06, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

cat("\nwriting figure 35...\n")
print(system.time(
  save_fig(f, "fig35_supervision_ladder", 8.2, 4.2)))
