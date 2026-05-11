# Phase 2 github-diff — paper 08 DP-Font (IJCAI 2024)

- **Paper**: Zhang, Zhu, Benarab, Ma, Dong, Sun. *DP-Font: Chinese Calligraphy Font Generation Using Diffusion Model and Physical Information Neural Network*. IJCAI 2024, pp. 7807-7815 (proceedings 0863).
- **Affiliation**: Harbin Engineering University, College of Computer Science and Technology.
- **Phase 0 note**: GitHub URL not recorded.
- **Phase 2 result**: `official_unavailable` — confirmed.

## 1. Code-availability investigation

Exhaustive search performed on 2026-05-11; **no official code release found**:

| Source | Query | Result |
|---|---|---|
| IJCAI proceedings page `proceedings/2024/863` | landing page links | only paper PDF link; no code/project page |
| IJCAI PDF `proceedings/2024/0863.pdf` | "code available at" / GitHub URL | text extraction failed (binary image-PDF) but landing page has no link |
| OpenReview `forum?id=SkrLT0tUq6` | "Code" / supplementary | no code, no supplementary materials linked |
| ACM Digital Library `10.24963/ijcai.2024/863` | code/data availability | none surfaced |
| ResearchGate publication 382788524 | code link | 403 (gated); no PwC mirror |
| paperswithcode `/paper/dp-font-...` | code repository tab | 302 redirect to `/papers/trending` — i.e., no paper-specific page registered, meaning **no code association on PwC** |
| `gh search repos "DP-Font"` (20 results) | name match | all results unrelated (font-awesome pickers, DPI font tools, etc.) |
| `gh search repos "DP-Font calligraphy"` / `"DP-Font PINN"` / `"calligraphy diffusion physical informed"` | topic match | 0 results |
| WebSearch `site:github.com "DP-Font" calligraphy diffusion` | site-restricted | 0 links |
| WebSearch author names (Liguo Zhang, Yalong Zhu) + github | author repo | only the IJCAI paper records returned; no personal repo |

No DP-Font implementation has been published by the authors, mirrored on a community fork, or registered on paperswithcode. Sources of leakage that *would* normally expose code (supplementary ZIP on IJCAI, OpenReview attachments, project README) all return empty.

## 2. Phase 3 implementation risk (unchanged → elevated)

Because no reference implementation exists, our reproduction depends entirely on the published text. The four areas that the Phase 2 brief flagged as diff targets cannot be diffed — they become Phase 3 **open risks**:

### 2.1 PINN loss formulation — HIGH risk

The paper claims to encode "the movement rule of the nib and the diffusion pattern of the ink" as physical-equation residuals added to the diffusion training loss, but the publicly accessible abstract / cover material does not include the closed-form equations. From the abstract on IJCAI and ResearchGate snippets we only know:

- two physical phenomena are modeled: **(a) nib trajectory dynamics**, **(b) ink diffusion on paper**;
- a physical-equation residual term is appended to the diffusion loss (the "P" in DP-Font);
- the paper reports an 8.48 % FID gain over Diff-Font, and an ablation `RAW / RAW+SO / RAW+PINN` (stroke order alone vs +PINN), implying PINN delivers a separable but unspecified loss term.

**No equation form is publicly available without paywalled / image-PDF access.** Our three surrogate physics models (Phase 1 designs) cannot be validated against the paper's actual formulation. Concretely:

- We do not know whether nib motion is modeled as a 2nd-order ODE (mass-spring-damper), a Hamiltonian, or a viscous-drag first-order ODE.
- We do not know whether ink diffusion is the heat / Fokker–Planck equation, an anisotropic diffusion (Perona–Malik) variant, or a Navier–Stokes-with-source formulation.
- We do not know the loss weighting (λ_PINN) — the paper reports only `RAW+PINN` as a switch.

**Mitigation**: in Phase 3 we should (a) attempt to obtain the camera-ready PDF with selectable text via the authors' personal page or library proxy, (b) treat our 3 surrogates as an *ablation family* rather than a reproduction, and (c) report results as "DP-Font-inspired" not "DP-Font reproduction".

### 2.2 "Dual-path" architecture — MEDIUM risk (naming risk)

The Phase 2 brief assumed a content-path / style-path dual encoder à la DM-Font. The IJCAI abstract describes the method as a **single diffusion model with multi-attribute guidance + PINN loss**, not a dual-path encoder. The "DP" in DP-Font almost certainly stands for **D**iffusion + **P**hysical-information, **not** "Dual-Path". This needs explicit confirmation from the full text before we mirror the architecture.

**Action**: re-read the Phase 1 brief and remove "dual path encoder" from our planned architecture if the paper does not in fact prescribe one.

### 2.3 Multi-attribute guidance — MEDIUM risk

Abstract says "multi-attribute guidance … introduces the critical constraint of stroke order". Public material does not enumerate the attribute set. Plausible candidates: `{character content, writer/style ID, stroke order sequence}`. We do not know whether script or source are conditioned attributes. Our `R_char + writer_id + unit_id` embedding stack is roughly parallel but lacks an explicit stroke-order channel.

### 2.4 Stroke order data source — MEDIUM risk

The paper introduces stroke order as a **conditioning feature**, not a derived signal. The public material gives no provenance. Plausible sources we could not confirm: **Make Me a Hanzi** (CC-BY MIT-licensed; covers ~9000 chars with ordered stroke SVGs), CASIA-OLHWDB (online handwriting, real human stroke order), or KanjiVG (Japanese, partial Chinese coverage).

Our current Phase 1 plan synthesises stroke order via SHA256-seeded permutation, which is **physically meaningless**. If DP-Font uses Make Me a Hanzi or CASIA, our reproduction is not faithful and our "stroke order" channel is decorative noise.

## 3. Things that did NOT change after Phase 2

- Phase 0 GitHub URL: still `not recorded` (correctly).
- We have not modified `src/`, configs, or training scripts (per instruction).
- No commit was made (per instruction).

## 4. Recommended Phase 3 next steps

1. Obtain a text-selectable copy of the camera-ready PDF (institutional library proxy or arXiv if mirrored) so we can pull the exact PINN equation form, λ weights, and stroke-order data citation.
2. Until then, freeze our 3 surrogate physics designs as an ablation **family**, not a "match".
3. Re-name the planned Phase 3 architecture from "dual-path" → "diffusion + PINN + multi-attribute guidance" to match what the paper actually claims.
4. Replace the SHA256-synthetic stroke order with Make Me a Hanzi sequences for any character covered by it (covers our 84-unit subset entirely for common chars) before we claim to reproduce the stroke-order constraint.
5. If after step 1 the equation form is still unrecoverable, write to the corresponding author (Jianguo Sun, Harbin Engineering University) for the loss expression — the paper makes a strong claim that is only verifiable from the formula.

## 5. Sources

- [DP-Font IJCAI proceedings page](https://www.ijcai.org/proceedings/2024/863)
- [DP-Font IJCAI PDF](https://www.ijcai.org/proceedings/2024/0863.pdf)
- [DP-Font OpenReview](https://openreview.net/forum?id=SkrLT0tUq6)
- [DP-Font ACM DL](https://dl.acm.org/doi/10.24963/ijcai.2024/863)
- [DP-Font ResearchGate (403)](https://www.researchgate.net/publication/382788524_DP-Font_Chinese_Calligraphy_Font_Generation_Using_Diffusion_Model_and_Physical_Information_Neural_Network)
