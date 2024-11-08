// Yoinked from https://nachtimwald.com/2017/11/18/base64-encode-and-decode-in-c/

static INV_TABLE: [isize; 80] = [
  62, -1, -1, -1, 63, 52, 53, 54, 55, 56, 57, 58,
	59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5,
	6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
	21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1, 26, 27, 28,
	29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
	43, 44, 45, 46, 47, 48, 49, 50, 51
];

fn calc_size(data_in: &[char]) -> usize {
	let mut ret = data_in.len() / 4 * 3;

	for x in data_in.iter().rev() {
		if *x == '=' {
			ret -= 1;
		} else {
			break;
		}
	}

	return ret;
}

fn is_valid_char(c: char) -> bool {
  return match c {
    '0'..='9' | 'A'..='Z' | 'a'..='z' | '+' | '/' | '=' => true,
    _ => false
  };
}

pub fn decode(data_in: &[char]) -> Option<Vec<u8>> {
  let mut buf = vec![0 as u8;calc_size(data_in)];

	if data_in.len() % 4 != 0 {
    return None;
  }

  if data_in.iter().any(|x|!is_valid_char(*x)) {
    return None;
  }

  let mut i = 0;
  let mut j = 0;

	while i < data_in.len() {
		let mut v = INV_TABLE[data_in[i] as usize - 43];
		v = (v << 6) | INV_TABLE[data_in[i+1] as usize - 43];
		v = if data_in[i+2] == '=' { v << 6 } else { (v << 6) | INV_TABLE[data_in[i+2] as usize - 43] };
		v = if data_in[i+3] == '=' { v << 6 } else { (v << 6) | INV_TABLE[data_in[i+3] as usize - 43] };

		buf[j] = ((v >> 16) & 0xff) as u8;

		if data_in[i+2] != '=' {
			buf[j+1] = ((v >> 8) & 0xFF) as u8;
    }

		if data_in[i+3] != '=' {
			buf[j+2] = (v & 0xFF) as u8;
    }

    i+=4;
    j+=3;
	}

	return Some(buf);
}