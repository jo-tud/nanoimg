//! Minimal BPE tokenizer for SigLIP2.
//! Parses tokenizer.json directly — no serde_json dependency.

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct BpeTokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    id_to_bytes: Vec<Vec<u8>>,
    merge_rank: HashMap<(u32, u32), u32>,
    pad_id: u32,
    unk_id: u32,
    eos_id: Option<u32>,
    /// Byte-fallback tokens `<0x00>`..`<0xFF>` for characters missing from the vocab
    /// (a byte without its own token maps to `<unk>`). Empty if the vocab has none.
    byte_ids: Vec<u32>,
}

impl BpeTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let (vocab_pairs, merges) = parse_tokenizer_json(&data)?;

        let mut vocab = HashMap::with_capacity(vocab_pairs.len());
        let max_id = vocab_pairs.iter().map(|(_, id)| *id).max().unwrap_or(0);
        let mut id_to_bytes = vec![Vec::new(); max_id as usize + 1];

        for (token, id) in &vocab_pairs {
            vocab.insert(token.clone(), *id);
            if (*id as usize) < id_to_bytes.len() {
                id_to_bytes[*id as usize] = token.clone();
            }
        }

        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, (left, right)) in merges.iter().enumerate() {
            if let (Some(&lid), Some(&rid)) = (vocab.get(left), vocab.get(right)) {
                merge_rank.insert((lid, rid), rank as u32);
            }
        }

        let pad_id = vocab.get(&b"<pad>"[..].to_vec()).copied().unwrap_or(0);
        let unk_id = vocab.get(&b"<unk>"[..].to_vec()).copied().unwrap_or(3);
        let eos_id = vocab.get(&b"<eos>"[..].to_vec()).copied();
        let byte_ids: Vec<Option<u32>> = (0..256)
            .map(|b| vocab.get(format!("<0x{b:02X}>").as_bytes()).copied())
            .collect();
        let byte_ids = if byte_ids.iter().any(Option::is_some) {
            byte_ids.into_iter().map(|id| id.unwrap_or(unk_id)).collect()
        } else {
            Vec::new()
        };

        Ok(Self { vocab, id_to_bytes, merge_rank, pad_id, unk_id, eos_id, byte_ids })
    }

    /// Encode `text` like the reference `tokenizers` pipeline for SigLIP2:
    /// BPE with byte fallback, then `<eos>`, then padding/truncation to `max_len`.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<i64> {
        // Normalize: replace space with ▁ (U+2581)
        let normalized = text.replace(' ', "\u{2581}");

        // Split into individual characters, map to vocab IDs.
        // Unknown characters fall back to their UTF-8 bytes (<0xNN> tokens).
        let mut ids: Vec<u32> = Vec::with_capacity(normalized.len());
        for c in normalized.chars() {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            match self.vocab.get(s.as_bytes()) {
                Some(&id) => ids.push(id),
                None if !self.byte_ids.is_empty() =>
                    ids.extend(s.bytes().map(|b| self.byte_ids[b as usize])),
                None => ids.push(self.unk_id),
            }
        }

        // Iterative BPE merging
        loop {
            if ids.len() < 2 { break; }
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;
            for i in 0..ids.len() - 1 {
                if let Some(&rank) = self.merge_rank.get(&(ids[i], ids[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }
            if best_rank == u32::MAX { break; }

            // Merge: concatenate the byte representations
            let left = &self.id_to_bytes[ids[best_pos] as usize];
            let right = &self.id_to_bytes[ids[best_pos + 1] as usize];
            let merged: Vec<u8> = [left.as_slice(), right.as_slice()].concat();
            let new_id = self.vocab.get(&merged).copied().unwrap_or(self.unk_id);
            ids[best_pos] = new_id;
            ids.remove(best_pos + 1);
        }

        // Truncate leaving room for <eos> (post-processor template: "$A <eos>"), then pad
        let mut result: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        if let Some(eos) = self.eos_id {
            result.truncate(max_len.saturating_sub(1));
            result.push(eos as i64);
        }
        result.truncate(max_len);
        result.resize(max_len, self.pad_id as i64);
        result
    }
}

// ── Minimal JSON parser ──────────────────────────────────────────────────────

struct JsonParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.data.len() && matches!(self.data[self.pos], b' ' | b'\n' | b'\r' | b'\t') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> u8 { self.data[self.pos] }
    fn advance(&mut self) { self.pos += 1; }
    fn expect(&mut self, c: u8) {
        assert_eq!(self.data[self.pos], c, "expected '{}' at pos {}", c as char, self.pos);
        self.pos += 1;
    }

    fn parse_string(&mut self) -> Vec<u8> {
        self.expect(b'"');
        let mut result = Vec::new();
        loop {
            let b = self.data[self.pos];
            self.pos += 1;
            match b {
                b'"' => return result,
                b'\\' => {
                    let next = self.data[self.pos];
                    self.pos += 1;
                    match next {
                        b'"' => result.push(b'"'),
                        b'\\' => result.push(b'\\'),
                        b'/' => result.push(b'/'),
                        b'n' => result.push(b'\n'),
                        b't' => result.push(b'\t'),
                        b'r' => result.push(b'\r'),
                        b'b' => result.push(0x08),
                        b'f' => result.push(0x0C),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.data[self.pos..self.pos + 4]).unwrap_or("0000");
                            self.pos += 4;
                            let cp = u32::from_str_radix(hex, 16).unwrap_or(0xFFFD);
                            if let Some(ch) = char::from_u32(cp) {
                                let mut buf = [0u8; 4];
                                let s = ch.encode_utf8(&mut buf);
                                result.extend_from_slice(s.as_bytes());
                            }
                        }
                        _ => { result.push(b'\\'); result.push(next); }
                    }
                }
                _ => result.push(b),
            }
        }
    }

    fn parse_int(&mut self) -> i64 {
        self.skip_ws();
        let neg = if self.peek() == b'-' { self.advance(); true } else { false };
        let mut val: i64 = 0;
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
            val = val * 10 + (self.data[self.pos] - b'0') as i64;
            self.pos += 1;
        }
        if neg { -val } else { val }
    }

    fn skip_value(&mut self) {
        self.skip_ws();
        match self.peek() {
            b'"' => { self.parse_string(); }
            b'{' => self.skip_object(),
            b'[' => self.skip_array(),
            b't' | b'f' | b'n' => {
                while self.pos < self.data.len() && self.data[self.pos].is_ascii_alphabetic() {
                    self.pos += 1;
                }
            }
            _ => {
                // number
                if self.peek() == b'-' { self.advance(); }
                while self.pos < self.data.len() && (self.data[self.pos].is_ascii_digit()
                    || matches!(self.data[self.pos], b'.' | b'e' | b'E' | b'+' | b'-'))
                {
                    self.pos += 1;
                }
            }
        }
    }

    fn skip_object(&mut self) {
        self.expect(b'{');
        self.skip_ws();
        if self.peek() == b'}' { self.advance(); return; }
        loop {
            self.skip_ws();
            self.parse_string(); // key
            self.skip_ws();
            self.expect(b':');
            self.skip_value();
            self.skip_ws();
            if self.peek() == b'}' { self.advance(); return; }
            self.expect(b',');
        }
    }

    fn skip_array(&mut self) {
        self.expect(b'[');
        self.skip_ws();
        if self.peek() == b']' { self.advance(); return; }
        loop {
            self.skip_value();
            self.skip_ws();
            if self.peek() == b']' { self.advance(); return; }
            self.expect(b',');
        }
    }
}

fn parse_tokenizer_json(data: &[u8]) -> Result<(Vec<(Vec<u8>, u32)>, Vec<(Vec<u8>, Vec<u8>)>)> {
    let mut p = JsonParser { data, pos: 0 };
    let mut vocab = Vec::new();
    let mut merges = Vec::new();

    p.skip_ws();
    p.expect(b'{');
    loop {
        p.skip_ws();
        if p.peek() == b'}' { break; }
        let key = p.parse_string();
        p.skip_ws();
        p.expect(b':');

        if key == b"model" {
            p.skip_ws();
            p.expect(b'{');
            loop {
                p.skip_ws();
                if p.peek() == b'}' { p.advance(); break; }
                let mkey = p.parse_string();
                p.skip_ws();
                p.expect(b':');

                if mkey == b"vocab" {
                    p.skip_ws();
                    p.expect(b'{');
                    p.skip_ws();
                    if p.peek() != b'}' {
                        loop {
                            p.skip_ws();
                            let token = p.parse_string();
                            p.skip_ws();
                            p.expect(b':');
                            let id = p.parse_int() as u32;
                            vocab.push((token, id));
                            p.skip_ws();
                            if p.peek() == b'}' { p.advance(); break; }
                            p.expect(b',');
                        }
                    } else {
                        p.advance();
                    }
                } else if mkey == b"merges" {
                    p.skip_ws();
                    p.expect(b'[');
                    p.skip_ws();
                    if p.peek() != b']' {
                        loop {
                            p.skip_ws();
                            if p.peek() == b'[' {
                                // Array-of-arrays format: [["a","b"], ...]
                                p.advance(); // [
                                p.skip_ws();
                                let left = p.parse_string();
                                p.skip_ws();
                                p.expect(b',');
                                p.skip_ws();
                                let right = p.parse_string();
                                p.skip_ws();
                                p.expect(b']');
                                merges.push((left, right));
                            } else {
                                // String format: ["a b", ...]
                                let s = p.parse_string();
                                if let Some(sp) = s.iter().position(|&b| b == b' ') {
                                    merges.push((s[..sp].to_vec(), s[sp + 1..].to_vec()));
                                }
                            }
                            p.skip_ws();
                            if p.peek() == b']' { p.advance(); break; }
                            p.expect(b',');
                        }
                    } else {
                        p.advance();
                    }
                } else {
                    p.skip_value();
                }

                p.skip_ws();
                if p.peek() == b',' { p.advance(); }
            }
        } else {
            p.skip_value();
        }

        p.skip_ws();
        if p.peek() == b',' { p.advance(); }
    }

    if vocab.is_empty() {
        bail!("no vocab found in tokenizer.json");
    }
    Ok((vocab, merges))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected IDs come from the reference HF `tokenizers` library
    /// (`Tokenizer.from_file(...)` with truncation to 64) on the same tokenizer.json.
    #[test]
    #[ignore] // needs ~/.nanoimg/models/tokenizer.json
    fn matches_reference_tokenizer() {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let tok = BpeTokenizer::load(&Path::new(&home).join(".nanoimg/models/tokenizer.json"))
            .expect("load tokenizer");
        let cases: &[(&str, &[i64])] = &[
            ("a photo of a cat", &[235250, 2686, 576, 476, 4401, 1]),
            ("zwei kinder spielen am strand", &[131201, 42900, 63859, 1144, 40672, 1]),
            ("🐶 dog", &[241878, 5929, 1]),
            // 𓀀 is not in the vocab → UTF-8 byte fallback
            ("𓀀 hieroglyph", &[457, 364, 345, 345, 186636, 836, 1]),
            ("  double  space ", &[139, 4576, 139, 11035, 235248, 1]),
        ];
        for (text, want) in cases {
            let got = tok.encode(text, 64);
            assert_eq!(got.len(), 64);
            assert_eq!(&got[..want.len()], *want, "text: {text:?}");
            assert!(got[want.len()..].iter().all(|&id| id == 0), "padding: {text:?}");
        }

        // Overlong input: truncated to 63 tokens, <eos> last
        let long = tok.encode(&["word"; 70].join(" "), 64);
        assert_eq!(long.len(), 64);
        assert_eq!(&long[..2], &[1928, 2204]);
        assert_eq!(&long[62..], &[2204, 1]);
    }
}
