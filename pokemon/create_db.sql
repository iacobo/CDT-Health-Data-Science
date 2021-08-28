DROP TABLE IF EXISTS Category;
CREATE TABLE Category (
  ID   INTEGER PRIMARY KEY AUTOINCREMENT,
  name NVARCHAR
);

DROP TABLE IF EXISTS Type;
CREATE TABLE Type (
  ID   INTEGER PRIMARY KEY AUTOINCREMENT,
  name NVARCHAR
);

DROP TABLE IF EXISTS Pokemon;
CREATE TABLE Pokemon (
  ID        INTEGER PRIMARY KEY AUTOINCREMENT,
  PokeNum   INT,
  name      NVARCHAR,
  type1ID   INT,
  type2ID   INT,
  habitat   NVARCHAR,
  gen       INT,
  capRate   FLOAT,
  genderPer FLOAT,
  evoName   NVARCHAR,
  mega      BOOL,
  catID     INT,
  nature    NVARCHAR,
  weight    FLOAT,
  height    FLOAT,
  bAtk      INT,
  bHp       INT,
  bSpd      INT,
  bSAtk     INT,
  bDef      INT,
  bSDef     INT,
  FOREIGN KEY (Type1ID) REFERENCES Type(ID),
  FOREIGN KEY (Type2ID) REFERENCES Type(ID),
  FOREIGN KEY (CatID)   REFERENCES Category(ID)
);

DROP TABLE IF EXISTS TypeMultiplier;
CREATE TABLE TypeMultiplier (
  atkTypeID INTEGER PRIMARY KEY,
  defType1  NVARCHAR,
  defType2  NVARCHAR,
  multi     FLOAT,
  FOREIGN KEY (atkTypeID) REFERENCES Type(ID)
);

DROP TABLE IF EXISTS Move;
CREATE TABLE Move (
  ID         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       NVARCHAR,
  PP         INT,
  typeID     INT,
  damage     INT,
  category   NVARCHAR,
  acc        INT,
  effectPC   INT,
  effectDesc NVARCHAR,
  FOREIGN KEY (typeID) REFERENCES TypeMultiplier(atkTypeID)
);

DROP TABLE IF EXISTS learnsMove;
CREATE TABLE learnsMove (
  moveID  INT,
  pokeID  INT,
  atLevel INT,
  viaItem BOOL,
  FOREIGN KEY (pokeID) REFERENCES Pokemon(ID)
  FOREIGN KEY (moveID) REFERENCES Move(ID)
);

DROP TABLE IF EXISTS EvoCondition;
CREATE TABLE EvoCondition (
  ID       INTEGER PRIMARY KEY AUTOINCREMENT,
  level    INT,
  item     NVARCHAR,
  affinity INT,
  time     INT,
  trade    NVARCHAR,
  other    BOOL
);

DROP TABLE IF EXISTS Evolution;
CREATE TABLE Evolution (
  prevolutionID INT,
  evolutionID   INT,
  conditionID   INT,
  FOREIGN KEY (PrevolutionID) REFERENCES Pokemon(ID),
  FOREIGN KEY (EvolutionID)   REFERENCES Pokemon(ID),
  FOREIGN KEY (ConditionID)   REFERENCES EvoCondition(ID)
);

DROP TABLE IF EXISTS Translations;
CREATE TABLE Translations (
  pokeID INT,
  german NVARCHAR,
  romaji NVARCHAR,
  FOREIGN KEY (pokeID) REFERENCES Pokemon(ID)
);
