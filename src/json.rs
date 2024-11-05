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

#[derive(Clone)]
enum Token {
  Eof,
  Char(char),
  String(String),
  Ident(String),
  Integer(u64),
  Boolean(bool)
}

struct Parser {
  text: Vec<char>,
  cur: usize,
  cache: Option<Token>
}

impl Parser {
  fn peek_char(&self) -> char {
    if self.cur < self.text.len() {
      return self.text[self.cur];
    }

    return '\0';
  }

  fn consume_char(&mut self) -> char {
    let result = self.peek_char();

    if self.cur < self.text.len() {
      self.cur += 1;
    }

    return result;
  }

  fn peek(&mut self) -> Result<Token, String> {
    if let None = self.cache {
      self.cache = Some(self.lex()?);
    }

    if let Some(tok) = &self.cache {
      return Ok(tok.clone());
    }
    else {
      return Err("Unreachable".to_string());
    }
  }

  fn is_ident(c: char) -> bool {
    return c.is_alphanumeric() || c == '_';
  }

  fn lex(&mut self) -> Result<Token, String> {
    if let Some(_) = self.cache {
      let result = self.cache.take();
      return Ok(result.unwrap());
    }

    while self.peek_char().is_whitespace() {
      self.consume_char();
    }

    let start = self.cur;
    let first = self.consume_char();

    return Ok( match first {
      '\0' => {
        Token::Eof
      }

      '"' => {
        while self.peek_char() != '"' && self.peek_char() != '\0' {
          self.consume_char();
        }

        if self.peek_char() != '"' {
          return Err("Unterminated string".to_string());
        }

        self.consume_char();

        Token::String(self.text[start+1..self.cur-1].iter().collect())
      }

      '0'..'9' => {
        while self.peek_char().is_digit(10) {
          self.consume_char();
        }

        let mut value = 0;

        for i in start..self.cur {
          value *= 10;
          value += self.text[i].to_digit(10).unwrap() as u64;
        }

        Token::Integer(value)
      }

      'a'..'z' | 'A'..'Z' | '_' => {
        while Self::is_ident(self.peek_char()) {
          self.consume_char();
        }

        let s: String = self.text[start..self.cur].iter().collect();

        if s == "true" {
          Token::Boolean(true)
        }
        else if s == "false" {
          Token::Boolean(false)
        }
        else {
          Token::Ident(s)
        }
      }
      
      _ => {
        Token::Char(first)
      }
    });
  }

  fn parse_number(&mut self) -> Result<Box<Node>, String> {
    if let Token::Integer(val) = self.lex()? {
      return Ok(Box::new(Node::Integer(val)));
    }
    else {
      return Err("Expected a number.".to_string());
    }
  }

  fn parse_string(&mut self) -> Result<Box<Node>, String> {
    if let Token::String(val) = self.lex()? {
      return Ok(Box::new(Node::String(val)));
    }
    else {
      return Err("Expected a string.".to_string());
    }
  }

  fn parse_boolean(&mut self) -> Result<Box<Node>, String> {
    if let Token::Boolean(val) = self.lex()? {
      return Ok(Box::new(Node::Boolean(val)));
    }
    else {
      return Err("Expected a boolean.".to_string());
    }
  }

  fn lex_char(&mut self) -> Result<char, String> {
    if let Token::Char(c) = self.peek()? {
      self.lex()?;
      return Ok(c);
    }
    else {
      return Err("Expected a character".to_string());
    }
  }

  fn parse_object(&mut self) -> Result<Box<Node>, String> {
    if self.lex_char()? != '{' {
      return Err("Expected an object".to_string());
    }

    let mut entries = HashMap::<String, Box<Node>>::new();

    loop {
      if let Token::Char(c) = self.peek()? {
        if c == '}' {
          break;
        }
      }

      if entries.len() > 0 {
        if let Token::Char(c) = self.lex()? {
          if c != ',' {
            return Err("Object entries must be separated by a ,".to_string());
          }
        }
        else {
          return Err("Object entries must be separated by a ,".to_string());
        }
      }

      match self.peek()? {
        Token::String(name) => {
          self.lex()?;

          if let Token::Char(c) = self.lex()? {
            if c != ':' {
              return Err("Expected a : after object entry name".to_string());
            }
          }
          else {
            return Err("Expected a : after object entry name".to_string());
          }

          let val = self.parse()?;

          entries.insert(name, val);
        }

        _ => {
          return Err("Expected a } or a object entry name".to_string());
        }
      }
    }

    if self.lex_char()? != '}' {
      return Err("Expected a }".to_string());
    }

    return Ok(Box::new(Node::Object(entries)));
  }

  fn parse_array(&mut self) -> Result<Box<Node>, String> {
    if self.lex_char()? != '[' {
      return Err("Expected an array".to_string());
    }

    let mut entries = Vec::<Box<Node>>::new();

    loop {
      if let Token::Char(c) = self.peek()? {
        if c == ']' {
          break;
        }
      }

      if entries.len() > 0 {
        if let Token::Char(c) = self.lex()? {
          if c != ',' {
            return Err("Array entries must be separated by a ,".to_string());
          }
        }
        else {
          return Err("Array entries must be separated by a ,".to_string());
        }
      }

      entries.push(self.parse()?);
    }

    if self.lex_char()? != ']' {
      return Err("Expected a ]".to_string());
    }

    return Ok(Box::new(Node::Array(entries)));
  }

  fn parse(&mut self) -> Result<Box<Node>, String> {
    match self.peek()? {
      Token::Eof => {
        return Err("End of file".to_string());
      }
      Token::Integer(_) => {
        return self.parse_number();
      },
      Token::String(_) => {
        return self.parse_string();
      },
      Token::Boolean(_) => {
        return self.parse_boolean();
      },
      Token::Ident(val) => {
        if val == "null" {
          self.lex()?;
          return Ok(Box::new(Node::Null));
        }
        else {
          return Err(format!("Invalid json character {}", val));
        }
      }
      Token::Char(c) => {
        match c {
          '{' => {
            return self.parse_object();
          }
          '[' => {
            return self.parse_array();
          }
          _ => {
            return Err(format!("Invalid json character {}", c));
          }
        }
      }
    }
  }
}

#[allow(unused)]
pub fn parse(text: &String) -> Result<Box<Node>, String> {
  let mut p = Parser {
    text: text.chars().collect(),
    cur: 0,
    cache: None
  };

  return p.parse();
}