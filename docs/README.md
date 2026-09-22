# Project website

This folder contains the static project website for:

> Benchmarking Models for Conversational E-Commerce:  
> A Reproducible Evaluation Framework

## Local review

From the repository root:

```bash
python3 -m http.server 4173 --directory docs
```

Open <http://127.0.0.1:4173/>.

Serving it locally is the primary review path because it reproduces GitHub Pages
more closely, loads local typography consistently, and enables the clipboard
interaction. The relative asset paths also permit direct opening from
`docs/index.html` where browser security settings allow it.

## Publication

The publication source is the `docs/` folder on the repository's default
branch. The live project page is:

<https://diogocarvalho88.github.io/conversational-commerce-benchmark-framework/>

The Open Graph cover is `assets/images/og-cover.png`; the page metadata uses
its absolute live URL for social previews.

## Content authority

Public claims are checked against:

1. the published camera-ready paper;
2. the files included in this public release;
3. the approved RecSys 2026 poster narrative.

The page intentionally excludes private system details, internal-model plans,
conversation traces beyond the example published in the paper and poster,
judge outputs, and unreleased product categories.

## Asset licenses

Barlow and Inter are distributed under the SIL Open Font License; license texts
are included in `assets/fonts/`. Paper figures and Rezolve identity assets are
used for this project page.
