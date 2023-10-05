CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY DEFAULT 0,
    student_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY DEFAULT 0,
    student_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);