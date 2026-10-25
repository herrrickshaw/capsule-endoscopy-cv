# Literature review: applicability and result types of small-bowel capsule endoscopy

Compiled 2026-10-25 for `capsule_cv`. Every figure below was read in the source document itself. The
"Source log" at the end records the route and depth (full text vs abstract) for each source. Numbers are
quoted as the authors report them. This is a review for engineering context, not clinical guidance.

---

## 1. Applicability: who gets a capsule study, and why

### Guideline indications (ESGE 2022 update)

The European Society of Gastrointestinal Endoscopy guideline [1] gives the current main recommendations
(MR) for small-bowel capsule endoscopy (SBCE):

| Indication | Recommendation | Strength / evidence |
|---|---|---|
| Suspected small-bowel bleeding | SBCE **first-line**, before other endoscopic and radiological tests (MR1) | strong / moderate |
| Overt bleeding | SBCE as soon as possible, **ideally within 48 h** (MR2) | strong / high |
| Before SBCE | No routine second-look endoscopy (MR3) | strong / low |
| Negative high-quality SBCE | Conservative management (MR4) | strong / moderate |
| Lesion found on SBCE | Device-assisted enteroscopy to confirm and treat (MR5) | strong / high |
| Iron-deficiency anaemia | SBCE first-line when small-bowel evaluation is indicated (MR6) | strong / high |
| Suspected Crohn's disease, negative ileocolonoscopy | SBCE as initial small-bowel test if no obstructive symptoms or known stenosis (MR7) | strong / high |
| Non-diagnostic cross-sectional imaging | SBCE if likely to change management (MR8) | strong / low |
| Established Crohn's disease | **Patency capsule first** to reduce retention (MR9) | strong / moderate |

The ESGE performance-measures paper [2] lists the full set of appropriate indications: obscure GI
bleeding, iron-deficiency anaemia, known or suspected Crohn's disease, small-bowel tumours, inherited
polyposis syndromes, abnormal radiological imaging, and complicated or refractory coeliac disease.

### What happens in practice (20-year meta-analysis)

Cortegoso Valdivia et al. [3] pooled 328 studies covering 86,930 patients. Obscure GI bleeding (OGIB)
was by far the most common indication (44,750 patients). Next came clinical symptoms (17,897), Crohn's
disease (11,299), neoplastic lesions (4,989) and coeliac disease (947).

| Indication | Detection rate | Completion rate | Retention rate |
|---|---|---|---|
| Obscure GI bleeding | 55% | 90.6% | 2% |
| Crohn's disease | 66% | 86.5% | 4% |
| Neoplastic lesions | 63% | 78.2% | 1% |
| Coeliac disease | 52% | 94.0% | 6% |
| Clinical symptoms | 62% | 92.8% | 2% |
| **All indications** | **59%** | **89.6%** | **2%** |

*(Table 4 of [3]. The abstract lists these values without row labels, so the table was used to assign them.)*

### Quality standards a capsule service is audited against (ESGE 2019)

There are six key performance measures for SBCE [2]:

| Measure | Minimum | Target |
|---|---|---|
| Appropriate indication | ≥95% | ≥95% |
| Complete caecal/stomal visualisation | ≥80% | ≥95% |
| Lesion detection rate (diagnostic yield per indication) | ≥50% | ≥50% |
| SBCE timing for overt bleeding | ≥90% | ≥90% |
| Appropriate referral for enteroscopy after SBCE | ≥75% | ≥90% |
| Capsule retention | <2% | <2% |

A 2025 update revised some benchmarks [4]. Only its abstract was accessible (see source log), so the
table above uses the 2019 values.

---

## 2. Types of scan results

### 2.1 Findings taxonomy (as labelled in Kvasir-Capsule)

Kvasir-Capsule [5] has 117 videos and 4,741,504 frames. Of those, 47,238 frames are expert-labelled with
bounding boxes across **14 classes**:

- **Anatomical landmarks:** pylorus, ileocecal valve, ampulla of Vater.
- **Lumen content:** normal clean mucosa, reduced mucosal view (stool, bubbles), blood – fresh,
  blood – hematin (small black stripes after minimal bleeding), foreign body (tablet residue, retained capsule).
- **Mucosal lesions:** erythema, angiectasia, erosion, ulcer, lymphangiectasia, polyp.

### 2.2 How findings are graded

**Bleeding potential (Saurin classification)** [6]: the ESGE position recommends this score for obscure
GI bleeding.

| Grade | Lesions | Bleeding potential |
|---|---|---|
| P0 | e.g. phlebectasia, diverticula without blood, nodules without mucosal break | none |
| P1 | red spots, small or isolated erosions, possibly diminutive angiectasias | low / uncertain |
| P2 | typical angiectasias, large ulcerations, tumours, varices | high |
| P3 | active bleeding (added later) | — |

In its validation, P2 lesions were graded identically by readers in 100% of cases, against 73% for P1
and 27% for P0. Therapeutic impact was 61% for P2 versus 23% for P0/P1 [6].

**Vascular lesions (Yano-Yamamoto)** [6]: originally devised for enteroscopy, later applied to capsule
findings.

- Type 1: angiectasia (1a, red spot <1 mm, and so on).
- Type 2: Dieulafoy's lesion.
- Type 3: arteriovenous malformation.

**Inflammation (Crohn's disease)** [6]:

- **Lewis score:** the most widely used index, built into the PillCam RAPID software. The small bowel is
  split into **tertiles by transit time**, and each tertile is scored on villous oedema, ulcers (number,
  longitudinal extent, size) and stenosis. The final score is the highest tertile.
  - <135: normal or clinically insignificant.
  - 135–<790: mild.
  - ≥790: moderate-to-severe.
  - A baseline score >350 predicted relapse within 6 months to 2 years.
- **CECDAI (Niv score):** proximal and distal halves, split at the small-bowel transit midpoint. Each half
  is rated on inflammation, extent and stricture. Its cut-offs relative to Lewis vary between studies.
  One prospective study found CECDAI <5.4 ≈ Lewis <135, and >9.2 ≈ Lewis >790.

**Masses and bulges (SPICE)** [6]: the SPICE score separates subepithelial masses from innocent bulges.
A score >2 gave 83.3% sensitivity and 86.4% specificity for a subepithelial mass in the derivation study.

**Cleansing and visualisation** [6]: the Park and Brotz scales are the two cited in the ESGE
performance-measures document. Automated measures exist too:

- The **computed assessment of cleansing (CAC)** score uses the **red/green (R/G) channel ratio** of the
  image colour bar.
- Oumrani et al. combined R/G colorimetry, bubble abundance and brightness. Against expert Brotz grading
  (≥7/10 adequate), this reached 90.0% sensitivity and 87.7% specificity.

---

## 3. Automated reading: what has been shown

| Study | Design | Result |
|---|---|---|
| **Ding et al. 2019** [7] | CNN trained on 158,235 images from 1,970 patients; validated on 5,000 patients across 77 centres | Abnormality sensitivity per patient: **99.88%** (CNN) vs **74.57%** (conventional reading). Mean reading time: **96.6 → 5.9 min** |
| **Xie et al. 2022** (JAMA Netw Open) [8] | 2,898 validation exams; AI-assisted reading ("SmartScan") vs conventional | Findings detected: **95.9% vs 76.1%**. Images to review per video: 27,911 → 779 (**96.1%** fewer). Reading time: **51.4 → 5.4 min** |
| **Bang et al. 2021** meta-analysis [9] | 39 computer-aided diagnosis studies | Haemorrhage/angioectasia: pooled sensitivity **0.96**, specificity **0.97**, AUC 0.99. Ulcer/erosion: sensitivity 0.93, specificity 0.92 |
| **Suspected Blood Indicator (SBI)** [10] | Vendor red-pixel flagging in RAPID Reader; review of prior studies plus a bench study | Overall lesion sensitivity **20–56.4%**, rising to **58.3–93%** with **active** bleeding. PPV 24–90.3%. Detection depended on background colour and capsule speed |

---

## 4. Mapping to `capsule_cv`

| Clinical result type | What the tool does now | What the literature says | Gap / next step |
|---|---|---|---|
| Active bleeding (Saurin **P3**, "blood – fresh") | Red-hue + R/G ≥ 2.2 pixel fraction → bleeding events | This is the same family as the vendor **SBI**, which is reliable mainly for *active* bleeding (58–93%) and weak overall (20–56%) [10]. Learned models reach about 0.96 / 0.97 [9] | Treat as an active-bleeding detector only. Validate on Kvasir "blood – fresh". Add a hematin (black-stripe) class |
| Angiectasia (Saurin P1/**P2**, Yano-Yamamoto 1) | `red` lesion: focal a* rise over local background | Angiectasia is the commonest bleeding source in OGIB, which is the top indication [3] | Add size-based P1/P2 grading (diminutive vs typical) |
| Erosion / ulcer (P1/**P2**; Lewis ulcers) | `pale` lesion: fibrin-like focal L* rise | Ulcer extent and number drive the Lewis score [6] | Count and size ulcers per tertile to give an approximate Lewis score |
| Region / landmarks (pylorus, ileocecal valve) | Colour/texture change points → boundaries, transit times | Lewis and CECDAI are defined on **transit-time** tertiles and halves [6]. Caecal reach is a key quality measure (≥80% / ≥95%) [2] | Report "caecum reached?" and tertile boundaries directly |
| Cleansing / visualisation | Brightness, glare, bile/debris, blur → "usable" | Automated R/G + bubbles + brightness matched Brotz grading at 90% / 87.7% [6] | Add a bubble detector. Output a Brotz-like per-segment adequacy grade |
| Reading-time reduction | Redundancy removal (22–86% on synthetic data) | AI-assisted reading cut images by 96% and reading time ~10–16× [7, 8] | Dedup alone won't reach that; it needs a learned "normal frame" filter |
| Polyps / tumours (P2, SPICE) | — | 4,989 patients in [3]; lowest completion rate (78.2%) | Not detected. Needs a learned detector (Kvasir has 8,262 labelled frames of this class) |
| Lymphangiectasia, erythema, villous atrophy (coeliac) | — | Kvasir classes; coeliac is a guideline indication with the highest retention (6%) [3] | Not detected |
| Foreign body / retention | — | Retention <2% is a key performance measure [2] | Could flag a static frame run at study end (a capsule that stopped moving) |

**Bottom line for the tool:** it covers the result types behind the most common indication (bleeding,
angiectasia, ulcers) plus quality and landmark timing. Its bleeding approach, however, is the classical
red-pixel method whose limits are well documented [10]. The literature's performance benchmarks come
from learned models trained on labelled data [7–9]. The obvious next step is evaluation, then
fine-tuning, on Kvasir-Capsule [5].

---

## References

1. Pennazio M, Rondonotti E, Despott EJ, et al. Small-bowel capsule endoscopy and device-assisted enteroscopy for diagnosis and treatment of small-bowel disorders: ESGE Guideline – Update 2022. *Endoscopy* 2023;55:58–95. [doi:10.1055/a-1973-3796](https://doi.org/10.1055/a-1973-3796)
2. Spada C, McNamara D, Despott EJ, et al. Performance measures for small-bowel endoscopy: an ESGE Quality Improvement Initiative. *United European Gastroenterol J* 2019;7:614–641. [doi:10.1177/2050640619850365](https://doi.org/10.1177/2050640619850365)
3. Cortegoso Valdivia P, Skonieczna-Żydecka K, Elosua A, et al. Indications, detection, completion and retention rates of capsule endoscopy in two decades of use: a systematic review and meta-analysis. *Diagnostics* 2022;12:1105. [doi:10.3390/diagnostics12051105](https://doi.org/10.3390/diagnostics12051105)
4. Sidhu R, Shiha MG, Carretero C, et al. Performance measures for small-bowel endoscopy: an ESGE Quality Improvement Initiative – Update 2025. *Endoscopy* 2025;57:366–389. [doi:10.1055/a-2522-1995](https://doi.org/10.1055/a-2522-1995)
5. Smedsrud PH, Thambawita V, Hicks SA, et al. Kvasir-Capsule, a video capsule endoscopy dataset. *Sci Data* 2021;8:142. [doi:10.1038/s41597-021-00920-z](https://doi.org/10.1038/s41597-021-00920-z)
6. Rosa B, Margalit-Yehuda R, Gatt K, et al. Scoring systems in clinical small-bowel capsule endoscopy: all you need to know! *Endosc Int Open* 2021;9:E802–E823. [doi:10.1055/a-1372-4051](https://doi.org/10.1055/a-1372-4051) (PMC8159625)
7. Ding Z, Shi H, Zhang H, et al. Gastroenterologist-level identification of small-bowel diseases and normal variants by capsule endoscopy using a deep-learning model. *Gastroenterology* 2019;157:1044–1054.e5. [doi:10.1053/j.gastro.2019.06.025](https://doi.org/10.1053/j.gastro.2019.06.025)
8. Xie X, Xiao YF, Zhao XY, et al. Development and validation of an artificial intelligence model for small bowel capsule endoscopy video review. *JAMA Netw Open* 2022;5:e2221992. [doi:10.1001/jamanetworkopen.2022.21992](https://doi.org/10.1001/jamanetworkopen.2022.21992)
9. Bang CS, Lee JJ, Baik GH. Computer-aided diagnosis of gastrointestinal ulcer and hemorrhage using wireless capsule endoscopy: systematic review and diagnostic test accuracy meta-analysis. *J Med Internet Res* 2021;23:e33267. [doi:10.2196/33267](https://doi.org/10.2196/33267)
10. Park SC, Chun HJ, Kim ES, et al. Sensitivity of the suspected blood indicator: an experimental study. *World J Gastroenterol* 2012;18:4169. (PMC3422798)

## Source log

| Ref | URL read | Route | Depth |
|---|---|---|---|
| 1 | portal.findresearcher.sdu.dk (University of Southern Denmark repository record) | Thieme returned a bot challenge, so the host was abandoned; OpenAlex alternate location used | **Abstract only** (all main recommendations MR1–MR11 quoted from it; body text and tables not read) |
| 2 | pmc.ncbi.nlm.nih.gov/articles/PMC6545716 | direct GET | full text |
| 3 | pmc.ncbi.nlm.nih.gov/articles/PMC9139699 | direct GET | full text (Table 4) |
| 4 | portal.findresearcher.sdu.dk (University of Southern Denmark repository record) | Thieme blocked; OpenAlex alternate location | **Abstract only**; revised benchmarks not read |
| 5 | pmc.ncbi.nlm.nih.gov/articles/PMC8160146 | direct GET | full text |
| 6 | pmc.ncbi.nlm.nih.gov/articles/PMC8159625 | direct GET (Thieme copy blocked) | full text |
| 7 | duomed.com hosted PDF of the published article | direct GET, pdftotext | full text |
| 8 | jamanetwork.com/journals/jamanetworkopen/fullarticle/2794207 | direct GET | full text |
| 9 | pmc.ncbi.nlm.nih.gov/articles/PMC8715364 | direct GET | full text |
| 10 | pmc.ncbi.nlm.nih.gov/articles/PMC3422798 | direct GET | full text |

Not used because only search snippets were available: ACG 2015 small-bowel bleeding guideline
(Gerson et al.), Soffer et al. 2020 deep-learning meta-analysis, and Cortegoso Valdivia et al. 2025
AI-reading meta-analysis. The original Saurin (2003), Lewis (2008/2012) and CECDAI (Gal 2008)
derivation papers were not read directly; their definitions and cut-offs are cited through the 2021
review [6].
