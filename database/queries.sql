/*
    by: Audrey Fellows
    name: queries.sql
    version: 0.0.0
    last modified: 10/7/2026
*/

-- Dev review queue
SELECT f.flag_id, f.video_id, v.file_path, f.user_comment, f.date_flagged
FROM flagged f JOIN video v ON v.video_id = f.video_id
WHERE f.status = 'pending'
ORDER BY f.date_flagged;

-- Training set: confirmed and not yet used
SELECT v.video_id, v.file_path
FROM flagged f JOIN video v ON v.video_id = f.video_id
WHERE f.status = 'confirmed' AND f.used_for_training = FALSE;