CREATE TABLE `my_attendance_table` (
  `n_id` int(11) NOT NULL AUTO_INCREMENT,
  `n_name` varchar(32) DEFAULT NULL,
  `n_update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`n_id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8;