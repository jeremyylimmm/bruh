use std::collections::HashMap;
use std::result::Result;

#[allow(unused)]
#[derive(Debug)]
pub enum Node {
  Null,
  Integer(u64),
  Real(f64),
  String(String),
  Boolean(bool),
  Array(Vec<Node>),
  Object(HashMap<String, Node>)
}


#[derive(Copy, Clone)]
enum TokenKind {
  Error,
  Eof,
  LBrace,
  RBrace,
  LSquare,
  RSquare,
  Comma,
  Colon,
  String,
  Integer,
  Real,
  True,
  False,
  Null,
}

#[derive(Copy, Clone)]
struct Token {
  kind: TokenKind,
  start: usize,
  end: usize,
  line: usize
}

struct Parser {
  text: Vec<char>,
  line: usize,
  cur: usize,
  cache: Option<Token>,
}

impl Parser {
  fn peekc_offset(&self, offset: usize) -> char {
    let index = self.cur + offset;

    if index < self.text.len() {
      return self.text[index];
    }

    return '\0';
  }

  fn peekc(&self) -> char {
    return self.peekc_offset(0);
  }

  fn match_string(&mut self, word: &str) -> bool {
    for (i, c) in word.chars().enumerate() {
      if self.peekc_offset(i) != c {
        return false;
      }
    }

    for _ in 0..word.len() {
      self.eatc();
    }

    return true;
  }

  fn eatc(&mut self) -> char {
    let c = self.peekc();

    if self.cur < self.text.len() {
      self.cur += 1;
    }

    return c;
  }

  fn peek(&mut self) -> Result<Token, String> {
    if let None = self.cache {
      let tok = self.lex()?;
      self.cache = Some(tok);
    }

    return Ok(self.cache.clone().unwrap());
  }

  fn lex(&mut self) -> Result<Token, String> {
    if let Some(_) = self.cache {
      return Ok(self.cache.take().unwrap());
    }

    loop {
      while self.peekc().is_whitespace() {
        if self.eatc() == '\n' {
          self.line += 1;
        }
      }

      if self.peekc() == '/' && self.peekc_offset(1) == '/' {
        while self.peekc() != '\n' && self.peekc() != '\0' {
          self.eatc();
        }
      }
      else {
        break;
      }
    }

    let start = self.cur;
    let first = self.eatc();
    let line = self.line;

    let kind = match first {
      '"' => {
        while self.peekc() != '"' && self.peekc() != '\0' {
          if self.eatc() == '\n' {
            self.line += 1;
          }
        }

        if self.peekc() != '"' {
          return Err(format!("Unterminated string on line {}", line));
        }

        self.eatc();

        TokenKind::String
      }
      't' => if self.match_string("rue") {
        TokenKind::True
      }
      else {
        TokenKind::Error
      }
      'f' => if self.match_string("alse") {
        TokenKind::False
      }
      else {
        TokenKind::Error
      }
      'n' => if self.match_string("ull") {
        TokenKind::Null
      }
      else {
        TokenKind::Error
      }
      '-' | '0'..='9' => {
        if self.peekc() == '-' {
          self.eatc();
        }

        while self.peekc().is_digit(10) {
          self.eatc();
        }

        let is_real = if self.peekc() == '.' {
          self.eatc();
          true
        }
        else {
          false
        };

        while self.peekc().is_digit(10) {
          self.eatc();
        }

        if is_real { TokenKind::Real } else { TokenKind::Integer }
      }
      ':' => TokenKind::Colon,
      ',' => TokenKind::Comma,
      '[' => TokenKind::LSquare,
      ']' => TokenKind::RSquare,
      '{' => TokenKind::LBrace,
      '}' => TokenKind::RBrace,
      '\0' => TokenKind::Eof,
      _ => TokenKind::Error,
    };

    return Ok(Token {
      kind,
      start,
      end: self.cur,
      line
    });
  }

  fn error(token: Token, msg: &str) -> Result<Node, String> {
    return Err(format!("Line {}: {}", token.line, msg));
  }

  fn parse_null(&mut self) -> Result<Node, String> {
    return if let TokenKind::Null = self.peek()?.kind {
      self.lex()?;
      Ok(Node::Null)
    }
    else {
      Self::error(self.peek()?, "expected 'null'")
    };
  }

  fn parse_string(&mut self) -> Result<Node, String> {
    return if let TokenKind::String = self.peek()?.kind {
      let tok = self.lex()?;
      Ok(Node::String(self.text[(tok.start+1)..(tok.end-1)].iter().collect()))
    }
    else {
      Self::error(self.peek()?, "expected a string")
    };
  }

  fn parse_boolean(&mut self) -> Result<Node, String> {
    return match self.peek()?.kind {
      TokenKind::True => {
        self.lex()?;
        Ok(Node::Boolean(true))
      },
      TokenKind::False => {
        self.lex()?;
        Ok(Node::Boolean(false))
      }
      _ => Self::error(self.peek()?, "expected 'true' or 'false'")
    };
  }

  fn parse_array(&mut self) -> Result<Node, String> {
    let lbrace = self.peek()?;

    if let TokenKind::LSquare = lbrace.kind {
      self.lex()?;
    }
    else {
      return Self::error(lbrace, "expected an array");
    }

    let mut array = Vec::<Node>::new();

    loop {
      match self.peek()?.kind {
        TokenKind::Eof | TokenKind::RSquare => {
          break;
        }
        _ => {}
      }

      if array.len() > 0 {
        if let TokenKind::Comma = self.peek()?.kind {
          self.lex()?;
        }
        else {
          return Self::error(self.peek()?, "missing comma between array entries");
        }
      }

      array.push(self.parse_any()?);
    }

    if let TokenKind::RSquare = self.peek()?.kind {
      self.lex()?;
    }
    else {
      return Self::error(lbrace, "no matching ']'");
    }

    return Ok(Node::Array(array));
  }

  fn parse_object(&mut self) -> Result<Node, String> {
    let lbrace = self.peek()?;

    if let TokenKind::LBrace = lbrace.kind {
      self.lex()?;
    }
    else {
      return Self::error(lbrace, "expected an array");
    }

    let mut dict = HashMap::<String, Node>::new();

    loop {
      match self.peek()?.kind {
        TokenKind::Eof | TokenKind::RBrace => {
          break;
        }
        _ => {}
      }

      if dict.len() > 0 {
        if let TokenKind::Comma = self.peek()?.kind {
          self.lex()?;
        }
        else {
          return Self::error(self.peek()?, "missing comma between object entries");
        }
      }

      let wrong_format_str = "object entries must be of the format <key> : <value>";

      let name: String = match self.peek()?.kind {
        TokenKind::String => {
          let tok = self.lex()?;
          self.text[tok.start+1..tok.end-1].iter().collect()
        }
        _ => {
          return Self::error(self.peek()?, &wrong_format_str);
        }
      };

      if let TokenKind::Colon = self.peek()?.kind {
        self.lex()?;
      }
      else {
          return Self::error(self.peek()?, &wrong_format_str);
      }

      let val = self.parse_any()?;

      dict.insert(name, val);
    }

    if let TokenKind::RBrace = self.peek()?.kind {
      self.lex()?;
    }
    else {
      return Self::error(lbrace, "no matching '}'");
    }

    return Ok(Node::Object(dict));
  }

  fn get_string(&self, tok: Token) -> String {
    return self.text[tok.start..tok.end].iter().collect();
  }

  fn parse_integer(&mut self) -> Result<Node, String> {
    return if let TokenKind::Integer = self.peek()?.kind {
      let tok = self.lex()?;
      let lexeme = self.get_string(tok);
      let val = lexeme.parse().unwrap();
      Ok(Node::Integer(val))
    }
    else {
      Self::error(self.peek()?, "expected an integer")
    };
  }

  fn parse_real(&mut self) -> Result<Node, String> {
    return if let TokenKind::Real = self.peek()?.kind {
      let tok = self.lex()?;
      let lexeme = self.get_string(tok);
      let val = lexeme.parse().unwrap();
      Ok(Node::Real(val))
    }
    else {
      Self::error(self.peek()?, "expected a floating-point number")
    }
  }

  fn parse_any(&mut self) -> Result<Node, String> {
    return match self.peek()?.kind {
      TokenKind::Null => self.parse_null(),
      TokenKind::True | TokenKind::False => self.parse_boolean(),
      TokenKind::Integer => self.parse_integer(),
      TokenKind::LBrace => self.parse_object(),
      TokenKind::LSquare => self.parse_array(),
      TokenKind::String => self.parse_string(),
      TokenKind::Real => self.parse_real(),
      _ => Self::error(self.peek()?, "unexpected token")
    }
  }
}

pub fn parse(text: &String) -> Result<Node, String> {
  let mut parser = Parser {
    text: text.chars().collect(),
    line: 1,
    cur: 0,
    cache: None
  };

  return parser.parse_any();
}