/*
    by: Audrey Fellows
    name: create_tables.sql
    version: 0.1.1
    last modified: 11/3/2026
*/

DROP TABLE flagged;
DROP TABLE video;
DROP TABLE owlbox;
DROP TABLE user_farm;
DROP TABLE farm;
DROP TABLE users;

CREATE TABLE users
(
    USER_ID             INTEGER GENERATED ALWAYS AS IDENTITY,
    first_name          VARCHAR2(50) NOT NULL,
    last_name           VARCHAR2(50) NOT NULL,
    email               VARCHAR2(255) UNIQUE NOT NULL,
    password_hash       VARCHAR2(255),
    role                VARCHAR2(6) DEFAULT 'farmer' NOT NULL
                        CHECK (role IN ('farmer', 'dev', 'admin')),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY         (user_id)
);

CREATE TABLE farm
(
    FARM_ID             CHAR(6) NOT NULL,
    street              VARCHAR2(50),
    city                VARCHAR2(50),
    zip                 VARCHAR2(10) NOT NULL,
    PRIMARY KEY         (farm_id) 
);

CREATE TABLE user_farm
(
    USER_ID             INTEGER NOT NULL,
    FARM_ID             CHAR(6) NOT NULL,
    joined_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY         (user_id, farm_id),
    FOREIGN KEY         (user_id) REFERENCES users (user_id),
    FOREIGN KEY         (farm_id) REFERENCES farm (farm_id)
);

CREATE TABLE owlbox
(
    OWLBOX_ID           CHAR(6) NOT NULL UNIQUE,
    farm_id             CHAR(6) NOT NULL,
    install_date        DATE NOT NULL,
    is_active           NUMBER(1) DEFAULT 1 NOT NULL
                        CHECK (is_active IN (0, 1)),
    PRIMARY KEY         (owlbox_id),
    FOREIGN KEY         (farm_id) REFERENCES farm (farm_id)
);

CREATE TABLE video
(
    VIDEO_ID            VARCHAR2(20) NOT NULL,
    date_start          DATE,
    start_time          TIMESTAMP,
    end_time            TIMESTAMP,
    owlbox_id           CHAR(6) NOT NULL,
    file_path           VARCHAR2(255) NOT NULL,
    PRIMARY KEY         (video_id),
    FOREIGN KEY         (owlbox_id) REFERENCES owlbox (owlbox_id)
);

CREATE TABLE flagged
(
    FLAG_ID             INTEGER GENERATED ALWAYS AS IDENTITY,
    video_id            VARCHAR2(20) NOT NULL UNIQUE,
    flagged_by          INTEGER NOT NULL,
    date_flagged        TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    status              VARCHAR2(10) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'confirmed', 'rejected')),
    reviewed_by         INTEGER,
    date_resolved       DATE,
    used_for_training   NUMBER(1) DEFAULT 0 NOT NULL
                        CHECK (used_for_training IN (0, 1)),
    PRIMARY KEY         (flag_id),
    FOREIGN KEY         (video_id) REFERENCES video (video_id),
    FOREIGN KEY         (flagged_by) REFERENCES users (user_id),
    FOREIGN KEY         (reviewed_by) REFERENCES users (user_id),
    CHECK               (used_for_training = 0 OR status = 'confirmed')
);

CREATE INDEX idx_owlbox_farm    ON owlbox (farm_id);
CREATE INDEX idx_video_owlbox   ON video (owlbox_id);
CREATE INDEX idx_flagged_status ON flagged (status);
