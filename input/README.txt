Inputs ACTUALLY used by this experiment (backend = roma).

  cup2.mp4         source video — the clip with the object to replace (cup).
  replacement.png  frame-0 reference — a Gemini edit of the video's first frame
                   with the NEW object (banana) placed where the OLD object (cup)
                   was. This is the ONLY generative input besides the video.

That's everything the pipeline consumes. From these two it GENERATES, from scratch:
  - per-frame SAM3 mask of the old object (cup) and new object (banana) on frame 0
  - per-frame EDIT region   = RoMa-propagated (banana∪cup) bbox   -> outputs/<name>/roma/masks
  - per-segment ANCHORS     = RoMa full-warp of replacement.png   -> outputs/<name>/roma/anchors
No prepared "assets" / bundle banana_masks are used.

Configured in ../config.yaml ; outputs go to editAnything/outputs/<name>/ .
