/*
    by: Audrey Fellows
    name: insert_test.sql
    version: 0.0.0
    last modified: 10/7/2026
*/

INSERT INTO farm (farm_id, street, city, zip) VALUES
('FRM001', '12 Barn Rd', 'Mammoth', '93546'),
('FRM002', '88 Orchard Ln', 'Bishop', '93514');

INSERT INTO users (first_name, last_name, email, password_hash, role) VALUES
('Ann',  'Lee',   'ann@example.com',  'fakehash1', 'farmer'),
('Bob',  'Ruiz',  'bob@example.com',  'fakehash2', 'farmer'),
('Cara', 'Dev',   'cara@example.com', 'fakehash3', 'dev');

INSERT INTO user_farm (user_id, farm_id) VALUES
(1, 'FRM001'), 
(2, 'FRM001'),   -- two users on one farm
(2, 'FRM002');   -- Bob also on a second farm

INSERT INTO owlbox (owlbox_id, farm_id, install_date) VALUES
('OWL001', 'FRM001', '2026-03-01'),
('OWL002', 'FRM001', '2026-03-15'),   -- two owlboxes on one farm
('OWL003', 'FRM002', '2026-04-02');

INSERT INTO video (video_id, owlbox_id, date_start, start_time, end_time, file_path) VALUES
('VID1', 'OWL001', '2026-05-01', '02:14:00', '02:14:30', '/videos/vid1.mp4'),
('VID2', 'OWL002', '2026-05-01', '03:40:00', '03:40:45', '/videos/vid2.mp4'),
('VID3', 'OWL003', '2026-05-02', '01:05:00', '01:05:20', '/videos/vid3.mp4');


-- FLAGGING TEST
-- A user flags a video
INSERT INTO flagged (video_id, flagged_by, user_comment)
VALUES ('VID1', 1, 'Just a shadow, no rodent');

-- Dev confirms it's a real false positive
UPDATE flagged
SET status = 'confirmed', reviewed_by = 3, date_resolved = CURRENT_TIMESTAMP
WHERE flag_id = 1;

-- Pipeline marks it as used
UPDATE flagged SET used_for_training = TRUE WHERE flag_id = 1;


-- BAD DATA THESE SHOULD FAIL
-- Duplicate flag by same user on same video (UNIQUE)
INSERT INTO flagged (video_id, flagged_by) VALUES ('VID1', 1);

-- Different user flags an already-flagged video
INSERT INTO flagged (video_id, flagged_by) VALUES ('VID1', 2);

-- Bad status value (CHECK)
UPDATE flagged SET status = 'maybe' WHERE flag_id = 1;

-- Video that doesn't exist (FOREIGN KEY)
INSERT INTO flagged (video_id, flagged_by) VALUES ('NOPE', 1);

-- Training flag on an unconfirmed flag (CHECK)
INSERT INTO flagged (video_id, flagged_by, used_for_training)
VALUES ('VID2', 1, TRUE);