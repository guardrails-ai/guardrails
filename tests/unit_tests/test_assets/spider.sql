PRAGMA foreign_keys=ON;
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "department" (
"Department_ID" int,
"Name" text,
"Creation" text,
"Ranking" int,
"Budget_in_Billions" real,
"Num_Employees" real,
PRIMARY KEY ("Department_ID")
);
INSERT INTO department VALUES(1,'State','1789','1',9.9600000000000008526,30265.999999999999999);
INSERT INTO department VALUES(2,'Treasury','1789','2',11.099999999999999644,115896.99999999999999);
INSERT INTO department VALUES(3,'Defense','1947','3',439.30000000000001135,3000000.0);
INSERT INTO department VALUES(4,'Justice','1870','4',23.399999999999998578,112556.99999999999999);
INSERT INTO department VALUES(5,'Interior','1849','5',10.699999999999999289,71436.000000000000002);
INSERT INTO department VALUES(6,'Agriculture','1889','6',77.599999999999994316,109831.99999999999999);
INSERT INTO department VALUES(7,'Commerce','1903','7',6.2000000000000001776,35999.999999999999999);
INSERT INTO department VALUES(8,'Labor','1913','8',59.700000000000002843,17346.999999999999999);
INSERT INTO department VALUES(9,'Health and Human Services','1953','9',543.20000000000004548,66999.999999999999998);
INSERT INTO department VALUES(10,'Housing and Urban Development','1965','10',46.200000000000002843,10599.999999999999999);
INSERT INTO department VALUES(11,'Transportation','1966','11',58.000000000000000001,58621.999999999999998);
INSERT INTO department VALUES(12,'Energy','1977','12',21.5,116099.99999999999999);
INSERT INTO department VALUES(13,'Education','1979','13',62.799999999999997156,4487.0000000000000001);
INSERT INTO department VALUES(14,'Veterans Affairs','1989','14',73.200000000000002842,234999.99999999999999);
INSERT INTO department VALUES(15,'Homeland Security','2002','15',44.600000000000001422,207999.99999999999999);
CREATE TABLE IF NOT EXISTS "head" (
"head_ID" int,
"name" text,
"born_state" text,
"age" real,
PRIMARY KEY ("head_ID")
);
INSERT INTO head VALUES(1,'Tiger Woods','Alabama',66.999999999999999998);
INSERT INTO head VALUES(2,'Sergio García','California',68.000000000000000001);
INSERT INTO head VALUES(3,'K. J. Choi','Alabama',69.0);
INSERT INTO head VALUES(4,'Dudley Hart','California',51.999999999999999998);
INSERT INTO head VALUES(5,'Jeff Maggert','Delaware',53.000000000000000001);
INSERT INTO head VALUES(6,'Billy Mayfair','California',69.0);
INSERT INTO head VALUES(7,'Stewart Cink','Florida',50.0);
INSERT INTO head VALUES(8,'Nick Faldo','California',55.999999999999999999);
INSERT INTO head VALUES(9,'Pádraig Harrington','Connecticut',43.000000000000000001);
INSERT INTO head VALUES(10,'Franklin Langham','Connecticut',66.999999999999999998);
CREATE TABLE IF NOT EXISTS "management" (
"department_ID" int,
"head_ID" int,
"temporary_acting" text,
PRIMARY KEY ("Department_ID","head_ID"),
FOREIGN KEY ("Department_ID") REFERENCES `department`("Department_ID"),
FOREIGN KEY ("head_ID") REFERENCES `head`("head_ID")
);
INSERT INTO management VALUES(2,5,'Yes');
INSERT INTO management VALUES(15,4,'Yes');
INSERT INTO management VALUES(2,6,'Yes');
INSERT INTO management VALUES(7,3,'No');
INSERT INTO management VALUES(11,10,'No');
COMMIT;