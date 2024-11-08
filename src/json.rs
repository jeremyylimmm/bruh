use std::collections::HashMap;
use std::result::Result;

#[allow(unused)]
pub enum Node {
  Null,
  Integer(u64),
  Real(f64),
  String(String),
  Boolean(bool),
  Array(Vec<Box<Node>>),
  Object(HashMap<String, Box<Node>>)
}

