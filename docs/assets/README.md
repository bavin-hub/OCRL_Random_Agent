# Figure sources

- `architecture/report-architecture.png`: Figure 2 from `../report/OCRL_Report.pdf`, page 2,
  rendered and cropped to the diagram for inline display.
- `architecture/autoregressive-gru.png`: rendered from the supplied `architecture/autoreg_gru.drawio-1.pdf`.
- `architecture/teacher-forcing-gru.png`: rendered from the supplied `architecture/teacher_forcing_gru.drawio-1.pdf`.
- `graphs/wm-itr_2500.pth_combined_motion.png` and
  `graphs/wm-itr_2500.pth_combined_grav_vel_tau.png`: original supplied plot images,
  displayed without alteration. These belong to the report experiment, not the
  subsequent checkpoint-7030/3710 evaluation.
- Assets prefixed `7030-` and `3710-`: recorded diagnostic results documented in
  `../validation.md`. Their graphs remain available but are not the featured
  README figures.

The original report and diagram PDFs are preserved. PNG exports allow GitHub to
show the architecture diagrams inline.

- `videos/pipeline.mp4` and its chapter stills: a reproducible methodology animation
  made by `scripts/create_pipeline_video.py`, using animated draw.io-style diagrams on white and baseline 1199 simulation footage.
- `videos/baseline-1199-rollout.mp4`: the separately trained MJLab baseline, evaluated
  with unmodified actor observations. Provenance is in `../baselines.json`.

The README embeds `videos/pipeline-preview.gif` for inline animation and links to
the MP4 with `?raw=true` to bypass GitHub's file viewer. Regenerate the preview
after updating the video (run from the repository root; requires FFmpeg):

```bash
ffmpeg -v error -y -i docs/assets/videos/pipeline.mp4 \
  -filter_complex 'fps=6,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle' \
  -loop 0 docs/assets/videos/pipeline-preview.gif
```

The preview preserves the full 53-second sequence at 6 fps; the MP4 retains the
original resolution and frame rate.
