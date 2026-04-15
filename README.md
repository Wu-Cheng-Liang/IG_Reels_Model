Batch fixed-M Bass training for multiple Instagram Reels.

Inputs
------
Dynamic:
  /Users/jimwu/Desktop/Bass_Model/Reels Data/reels_dynamic_info.csv
  Expected columns:
    reels_shortcode,views,plays,likes,comments,timestamp

Static:
  /Users/jimwu/Desktop/Bass_Model/Reels Data/reels_static_info.csv
  Expected columns:
    kol_account,reels_shortcode,post_time,duration,caption

Selection rules
---------------
- Skip rows with missing view data.
- Keep reels with static post_time on/after 2026-04-10.
- Rank by usable data points after cleaning.
- Train at most 20 reels.

Outputs
-------
For each reel:
  /Users/jimwu/Desktop/Bass_Model/Output/<reels_shortcode>/
    - observed_vs_fit.csv
    - 30d_projection.csv
    - effective_M.csv
    - summary.json
    - fixed_M_bass_fit_30d.png
    - effective_M.png
    - effective_M_log.png

Global outputs:
  /Users/jimwu/Desktop/Bass_Model/Output/
    - selected_reels.csv
    - training_summary.csv
    - training_errors.csv
