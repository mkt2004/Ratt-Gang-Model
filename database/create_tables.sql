/*
    by: Audrey Fellows
    name: create_tables.sql
    version: 0.0.0
    last modified: 10/7/2026
*/

DROP TABLE IF EXISTS flagged;
DROP TABLE IF EXISTS video;
DROP TABLE IF EXISTS owlbox;
DROP TABLE IF EXISTS user_farm;
DROP TABLE IF EXISTS farm;
DROP TABLE IF EXISTS user;

CREATE TABLE users
(
    USER_ID             INTEGER GENERATED ALWAYS AS IDENTITY,
    first_name          VARCHAR(50) NOT NULL,
    last_name           VARCHAR(50) NOT NULL,
    email               VARCHAR(255) NOT NULL UNIQUE,
    password_hash       VARCHAR(255),
    role                VARCHAR(6) NOT NULL DEFAULT 'farmer'
                        CHECK (role IN ('farmer', 'dev', 'admin')),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY         (user_id)
);

CREATE TABLE farm
(
    FARM_ID             CHAR(6) NOT NULL UNIQUE,
    street              VARCHAR(50),
    city                VARCHAR(50),
    zip                 VARCHAR(10) NOT NULL,
    PRIMARY KEY         (farm_id) 
);

CREATE TABLE user_farm
(
    USER_ID             INTEGER NOT NULL,
    FARM_ID             CHAR(6) NOT NULL,
    joined_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY         (user_id, farm_id),
    FOREIGN KEY         (user_id) REFERENCES users,
    FOREIGN KEY         (farm_id) REFERENCES farm
)

CREATE TABLE owlbox
(
    OWLBOX_ID           CHAR(6) NOT NULL UNIQUE,
    farm_id             CHAR(6) NOT NULL,
    install_date        DATE NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY         (owlbox_id),
    FOREIGN KEY         (farm_id) REFERENCES farm
);

CREATE TABLE video
(
    VIDEO_ID            VARCHAR(20) NOT NULL UNIQUE,
    date_start          DATE,
    start_time          TIME,
    end_time            TIME,
    owlbox_id           CHAR(6) NOT NULL,
    file_path           VARCHAR(255) NOT NULL,
    PRIMARY KEY         (video_id),
    FOREIGN KEY         (owlbox_id) REFERENCES owlbox
);

CREATE TABLE flagged
(
    FLAG_ID             INTEGER GENERATED ALWAYS AS IDENTITY,
    video_id            CHAR(20) NOT NULL UNIQUE,
    flagged_by          INTEGER NOT NULL,
    date_flagged        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_comment        VARCHAR(500),
    status              VARCHAR(10) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'confirmed', 'rejected')),
    reviewed_by         INTEGER,
    date_resolved       DATE,
    used_for_training   BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY         (flag_id),
    FOREIGN KEY         (video_id) REFERENCES video,
    FOREIGN KEY         (flagged_by) REFERENCES users (user_id),
    FOREIGN KEY         (reviewed_by) REFERENCES users (user_id),
    CHECK               (used_for_training = FALSE OR status = 'confirmed')
);

CREATE INDEX idx_owlbox_farm    ON owlbox (farm_id);
CREATE INDEX idx_video_owlbox   ON video (owlbox_id);
CREATE INDEX idx_flagged_status ON flagged (status);