use anyhow::Result;
use image::imageops::FilterType;
use image::DynamicImage;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

// ── Constants ───────────────────────────────────────────────────────────────

const BG: u32 = 0x001a1a1a;
const SEL_COLOR: u32 = 0x0000aaff;
const BAR_BG: u32 = 0x00282828;
const BAR_FG: u32 = 0x00cccccc;
const SEL_BORDER: usize = 3;
const BAR_HEIGHT: usize = 24;
const MAX_DIM: u32 = 2048;
const SPACING: usize = 4;
const LOAD_BG: u32 = 0x00252525;
const LOAD_FG: u32 = 0x00666666;
const PREFETCH_ROWS: usize = 2;

// ── Bitmap font (5x7, ASCII 32–126) ────────────────────────────────────────

const FONT_W: usize = 5;
const FONT_H: usize = 7;

#[rustfmt::skip]
const FONT: [[u8; 7]; 95] = [
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00],[0x04,0x04,0x04,0x04,0x04,0x00,0x04],
    [0x0a,0x0a,0x00,0x00,0x00,0x00,0x00],[0x0a,0x1f,0x0a,0x0a,0x1f,0x0a,0x00],
    [0x04,0x0f,0x14,0x0e,0x05,0x1e,0x04],[0x19,0x1a,0x02,0x04,0x0b,0x13,0x00],
    [0x08,0x14,0x14,0x08,0x15,0x12,0x0d],[0x04,0x04,0x00,0x00,0x00,0x00,0x00],
    [0x02,0x04,0x08,0x08,0x08,0x04,0x02],[0x08,0x04,0x02,0x02,0x02,0x04,0x08],
    [0x00,0x0a,0x04,0x1f,0x04,0x0a,0x00],[0x00,0x04,0x04,0x1f,0x04,0x04,0x00],
    [0x00,0x00,0x00,0x00,0x00,0x04,0x08],[0x00,0x00,0x00,0x1f,0x00,0x00,0x00],
    [0x00,0x00,0x00,0x00,0x00,0x00,0x04],[0x01,0x02,0x02,0x04,0x08,0x08,0x10],
    [0x0e,0x11,0x13,0x15,0x19,0x11,0x0e],[0x04,0x0c,0x04,0x04,0x04,0x04,0x0e],
    [0x0e,0x11,0x01,0x06,0x08,0x10,0x1f],[0x0e,0x11,0x01,0x06,0x01,0x11,0x0e],
    [0x02,0x06,0x0a,0x12,0x1f,0x02,0x02],[0x1f,0x10,0x1e,0x01,0x01,0x11,0x0e],
    [0x06,0x08,0x10,0x1e,0x11,0x11,0x0e],[0x1f,0x01,0x02,0x04,0x08,0x08,0x08],
    [0x0e,0x11,0x11,0x0e,0x11,0x11,0x0e],[0x0e,0x11,0x11,0x0f,0x01,0x02,0x0c],
    [0x00,0x00,0x04,0x00,0x00,0x04,0x00],[0x00,0x00,0x04,0x00,0x00,0x04,0x08],
    [0x02,0x04,0x08,0x10,0x08,0x04,0x02],[0x00,0x00,0x1f,0x00,0x1f,0x00,0x00],
    [0x08,0x04,0x02,0x01,0x02,0x04,0x08],[0x0e,0x11,0x01,0x06,0x04,0x00,0x04],
    [0x0e,0x11,0x17,0x15,0x17,0x10,0x0e],[0x0e,0x11,0x11,0x1f,0x11,0x11,0x11],
    [0x1e,0x11,0x11,0x1e,0x11,0x11,0x1e],[0x0e,0x11,0x10,0x10,0x10,0x11,0x0e],
    [0x1e,0x11,0x11,0x11,0x11,0x11,0x1e],[0x1f,0x10,0x10,0x1e,0x10,0x10,0x1f],
    [0x1f,0x10,0x10,0x1e,0x10,0x10,0x10],[0x0e,0x11,0x10,0x17,0x11,0x11,0x0f],
    [0x11,0x11,0x11,0x1f,0x11,0x11,0x11],[0x0e,0x04,0x04,0x04,0x04,0x04,0x0e],
    [0x07,0x02,0x02,0x02,0x02,0x12,0x0c],[0x11,0x12,0x14,0x18,0x14,0x12,0x11],
    [0x10,0x10,0x10,0x10,0x10,0x10,0x1f],[0x11,0x1b,0x15,0x15,0x11,0x11,0x11],
    [0x11,0x19,0x15,0x13,0x11,0x11,0x11],[0x0e,0x11,0x11,0x11,0x11,0x11,0x0e],
    [0x1e,0x11,0x11,0x1e,0x10,0x10,0x10],[0x0e,0x11,0x11,0x11,0x15,0x12,0x0d],
    [0x1e,0x11,0x11,0x1e,0x14,0x12,0x11],[0x0e,0x11,0x10,0x0e,0x01,0x11,0x0e],
    [0x1f,0x04,0x04,0x04,0x04,0x04,0x04],[0x11,0x11,0x11,0x11,0x11,0x11,0x0e],
    [0x11,0x11,0x11,0x11,0x0a,0x0a,0x04],[0x11,0x11,0x11,0x15,0x15,0x1b,0x11],
    [0x11,0x11,0x0a,0x04,0x0a,0x11,0x11],[0x11,0x11,0x0a,0x04,0x04,0x04,0x04],
    [0x1f,0x01,0x02,0x04,0x08,0x10,0x1f],[0x0e,0x08,0x08,0x08,0x08,0x08,0x0e],
    [0x10,0x08,0x08,0x04,0x02,0x02,0x01],[0x0e,0x02,0x02,0x02,0x02,0x02,0x0e],
    [0x04,0x0a,0x11,0x00,0x00,0x00,0x00],[0x00,0x00,0x00,0x00,0x00,0x00,0x1f],
    [0x08,0x04,0x00,0x00,0x00,0x00,0x00],[0x00,0x00,0x0e,0x01,0x0f,0x11,0x0f],
    [0x10,0x10,0x1e,0x11,0x11,0x11,0x1e],[0x00,0x00,0x0e,0x11,0x10,0x11,0x0e],
    [0x01,0x01,0x0f,0x11,0x11,0x11,0x0f],[0x00,0x00,0x0e,0x11,0x1f,0x10,0x0e],
    [0x06,0x08,0x1c,0x08,0x08,0x08,0x08],[0x00,0x00,0x0f,0x11,0x0f,0x01,0x0e],
    [0x10,0x10,0x1e,0x11,0x11,0x11,0x11],[0x04,0x00,0x0c,0x04,0x04,0x04,0x0e],
    [0x02,0x00,0x06,0x02,0x02,0x12,0x0c],[0x10,0x10,0x12,0x14,0x18,0x14,0x12],
    [0x0c,0x04,0x04,0x04,0x04,0x04,0x0e],[0x00,0x00,0x1a,0x15,0x15,0x11,0x11],
    [0x00,0x00,0x1e,0x11,0x11,0x11,0x11],[0x00,0x00,0x0e,0x11,0x11,0x11,0x0e],
    [0x00,0x00,0x1e,0x11,0x1e,0x10,0x10],[0x00,0x00,0x0f,0x11,0x0f,0x01,0x01],
    [0x00,0x00,0x16,0x19,0x10,0x10,0x10],[0x00,0x00,0x0f,0x10,0x0e,0x01,0x1e],
    [0x08,0x08,0x1c,0x08,0x08,0x09,0x06],[0x00,0x00,0x11,0x11,0x11,0x11,0x0f],
    [0x00,0x00,0x11,0x11,0x11,0x0a,0x04],[0x00,0x00,0x11,0x11,0x15,0x15,0x0a],
    [0x00,0x00,0x11,0x0a,0x04,0x0a,0x11],[0x00,0x00,0x11,0x11,0x0f,0x01,0x0e],
    [0x00,0x00,0x1f,0x02,0x04,0x08,0x1f],[0x02,0x04,0x04,0x08,0x04,0x04,0x02],
    [0x04,0x04,0x04,0x04,0x04,0x04,0x04],[0x08,0x04,0x04,0x02,0x04,0x04,0x08],
    [0x00,0x00,0x08,0x15,0x02,0x00,0x00],
];

fn draw_text(buf: &mut [u32], buf_w: usize, x: usize, y: usize, text: &str, fg: u32, bg: u32) {
    let buf_h = buf.len() / buf_w.max(1);
    let mut cx = x;
    for ch in text.chars() {
        let idx = (ch as usize).wrapping_sub(32);
        if idx >= 95 { cx += FONT_W + 1; continue; }
        let glyph = &FONT[idx];
        for row in 0..FONT_H {
            let py = y + row;
            if py >= buf_h { break; }
            for col in 0..FONT_W {
                let px = cx + col;
                if px >= buf_w { break; }
                let bit = (glyph[row] >> (4 - col)) & 1;
                buf[py * buf_w + px] = if bit == 1 { fg } else { bg };
            }
        }
        cx += FONT_W + 1;
    }
}

fn draw_placeholder(buf: &mut [u32], buf_w: usize, buf_h: usize,
                    x: usize, y: isize, w: usize, h: usize, frame: usize) {
    for row in 0..h {
        let dy = y + row as isize;
        if dy < 0 { continue; }
        let dy = dy as usize;
        if dy >= buf_h { break; }
        for col in 0..w {
            let dx = x + col;
            if dx >= buf_w { continue; }
            buf[dy * buf_w + dx] = LOAD_BG;
        }
    }
    let text = match frame % 4 {
        0 => "Loading   ", 1 => "Loading.  ", 2 => "Loading.. ", _ => "Loading...",
    };
    let tw = text.len() * (FONT_W + 1);
    let tx = x + w.saturating_sub(tw) / 2;
    let text_y = y + (h as isize - FONT_H as isize) / 2;
    if text_y >= 0 && (text_y as usize) < buf_h {
        draw_text(buf, buf_w, tx, text_y as usize, &text, LOAD_FG, LOAD_BG);
    }
}

// ── Cached thumbnail ────────────────────────────────────────────────────────

struct CachedThumb {
    pixels: Vec<u32>,
    w: usize,
    h: usize,
}

fn rgb_to_buf(rgb: &image::RgbImage) -> CachedThumb {
    let pixels: Vec<u32> = rgb.pixels()
        .map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32)
        .collect();
    CachedThumb { pixels, w: rgb.width() as usize, h: rgb.height() as usize }
}

/// Resize to fill cell exactly (scale to cover + center crop). For grid cells.
fn render_fill(img: &DynamicImage, w: usize, h: usize) -> CachedThumb {
    rgb_to_buf(&img.resize_to_fill(w as u32, h as u32, FilterType::Triangle).to_rgb8())
}

/// Resize to fit within bounds (aspect-preserving, no crop). For fullscreen.
fn render_fit(img: &DynamicImage, max_w: usize, max_h: usize) -> CachedThumb {
    rgb_to_buf(&img.resize(max_w as u32, max_h as u32, FilterType::Triangle).to_rgb8())
}

// ── Justified layout ────────────────────────────────────────────────────────

#[derive(Clone)]
struct Rect { x: usize, y: usize, w: usize, h: usize }

struct JustifiedLayout {
    cells: Vec<Rect>,
    row_ranges: Vec<(usize, usize)>, // (start, end) item indices per row
    item_to_row: Vec<usize>,
}

fn compute_layout(aspects: &[f64], container_w: usize, viewport_h: usize) -> JustifiedLayout {
    let n = aspects.len();
    if n == 0 {
        return JustifiedLayout { cells: vec![], row_ranges: vec![], item_to_row: vec![] };
    }

    let target_h = (viewport_h / 3).clamp(150, 400) as f64;
    let mut cells = vec![Rect { x: 0, y: 0, w: 0, h: 0 }; n];
    let mut row_ranges = vec![];
    let mut item_to_row = vec![0usize; n];
    let mut y = 0usize;
    let mut row_start = 0usize;

    for i in 0..n {
        let count = i - row_start + 1;
        let total_spacing = if count > 1 { (count - 1) * SPACING } else { 0 };
        let usable_w = container_w.saturating_sub(total_spacing) as f64;
        let sum_r: f64 = aspects[row_start..=i].iter().sum();
        let row_h = usable_w / sum_r;

        let is_last = i == n - 1;
        if row_h <= target_h || is_last {
            let actual_h = if is_last && row_h > target_h {
                target_h as usize
            } else {
                row_h as usize
            };
            let actual_h = actual_h.max(1);

            let mut x = 0usize;
            for j in row_start..=i {
                let cell_w = (aspects[j] * actual_h as f64).round() as usize;
                cells[j] = Rect { x, y, w: cell_w, h: actual_h };
                item_to_row[j] = row_ranges.len();
                x += cell_w + SPACING;
            }

            // Fix last cell to fill width exactly (only if row is justified)
            if !is_last || row_h <= target_h {
                let last_x = cells[i].x;
                cells[i].w = container_w.saturating_sub(last_x);
            }

            row_ranges.push((row_start, i + 1));
            y += actual_h + SPACING;
            row_start = i + 1;
        }
    }

    JustifiedLayout { cells, row_ranges, item_to_row }
}

// ── Items (lazy-loaded) ─────────────────────────────────────────────────────

struct Item {
    path: String,
    score: f64,
    aspect: f64,
    img: Option<DynamicImage>,
    thumb: Option<CachedThumb>,
    full: Option<CachedThumb>,
}

impl Item {
    fn ensure_img(&mut self) {
        if self.img.is_some() { return; }
        let mut img = image::open(&self.path).unwrap_or_else(|_| DynamicImage::new_rgb8(1, 1));
        if img.width() > MAX_DIM || img.height() > MAX_DIM {
            img = img.resize(MAX_DIM, MAX_DIM, FilterType::Triangle);
        }
        self.img = Some(img);
    }
}

fn make_items(results: &[(f64, String)]) -> Vec<Item> {
    results.iter().map(|(score, path)| {
        let aspect = image::image_dimensions(path)
            .map(|(w, h)| w as f64 / h.max(1) as f64)
            .unwrap_or(1.5);
        Item { path: path.clone(), score: *score, aspect, img: None, thumb: None, full: None }
    }).collect()
}

// ── Blitting helpers ────────────────────────────────────────────────────────

fn blit(buf: &mut [u32], buf_w: usize, buf_h: usize, thumb: &CachedThumb, tx: usize, ty: isize) {
    for row in 0..thumb.h {
        let dy = ty + row as isize;
        if dy < 0 { continue; }
        let dy = dy as usize;
        if dy >= buf_h { break; }
        let copy_w = thumb.w.min(buf_w.saturating_sub(tx));
        if copy_w == 0 { continue; }
        let src = row * thumb.w;
        let dst = dy * buf_w + tx;
        buf[dst..dst + copy_w].copy_from_slice(&thumb.pixels[src..src + copy_w]);
    }
}

fn draw_selection(buf: &mut [u32], buf_w: usize, buf_h: usize,
                  rx: usize, ry: isize, rw: usize, rh: usize, color: u32) {
    for t in 0..SEL_BORDER {
        // Top border
        let py = ry + t as isize;
        if py >= 0 && (py as usize) < buf_h {
            for x in rx..((rx + rw).min(buf_w)) {
                buf[py as usize * buf_w + x] = color;
            }
        }
        // Bottom border
        let py = ry + rh as isize - 1 - t as isize;
        if py >= 0 && (py as usize) < buf_h {
            for x in rx..((rx + rw).min(buf_w)) {
                buf[py as usize * buf_w + x] = color;
            }
        }
    }
    // Left/right borders
    for row in SEL_BORDER..(rh.saturating_sub(SEL_BORDER)) {
        let py = ry + row as isize;
        if py < 0 { continue; }
        let py = py as usize;
        if py >= buf_h { break; }
        for t in 0..SEL_BORDER {
            if rx + t < buf_w { buf[py * buf_w + rx + t] = color; }
            let right = rx + rw - 1 - t;
            if right < buf_w { buf[py * buf_w + right] = color; }
        }
    }
}

fn draw_bar(buf: &mut [u32], w: usize, h: usize, text: &str) {
    let bar_y = h.saturating_sub(BAR_HEIGHT);
    for y in bar_y..h {
        for x in 0..w { buf[y * w + x] = BAR_BG; }
    }
    let text_y = bar_y + (BAR_HEIGHT.saturating_sub(FONT_H)) / 2;
    draw_text(buf, w, 8, text_y, text, BAR_FG, BAR_BG);
}

// ── Viewer state ────────────────────────────────────────────────────────────

enum ViewState { Grid, Full }

struct Viewer {
    items: Vec<Item>,
    sel: usize,
    state: ViewState,
    layout: JustifiedLayout,
    scroll_y: usize,
    win_w: usize,
    win_h: usize,
    buf: Vec<u32>,
    dirty: bool,
    quit: bool,
    frame: usize,
}

impl Viewer {
    fn new(results: &[(f64, String)], w: usize, h: usize) -> Self {
        let items = make_items(results);
        let viewport_h = h.saturating_sub(BAR_HEIGHT);
        let aspects: Vec<f64> = items.iter().map(|i| i.aspect).collect();
        let layout = compute_layout(&aspects, w, viewport_h);
        Self {
            items, sel: 0, state: ViewState::Grid, layout,
            scroll_y: 0, win_w: w, win_h: h, buf: vec![BG; w * h],
            dirty: true, quit: false, frame: 0,
        }
    }

    fn recompute_layout(&mut self) {
        let viewport_h = self.viewport_h();
        let aspects: Vec<f64> = self.items.iter().map(|i| i.aspect).collect();
        self.layout = compute_layout(&aspects, self.win_w, viewport_h);
        for item in &mut self.items { item.thumb = None; }
    }

    fn resize(&mut self, w: usize, h: usize) {
        if w == 0 || h == 0 { return; }
        self.win_w = w;
        self.win_h = h;
        self.buf.resize(w * h, BG);
        self.recompute_layout();
        for item in &mut self.items { item.full = None; }
        self.ensure_sel_visible();
        self.dirty = true;
    }

    fn viewport_h(&self) -> usize {
        self.win_h.saturating_sub(BAR_HEIGHT)
    }

    fn content_h(&self) -> usize {
        if let Some(&(s, _)) = self.layout.row_ranges.last() {
            self.layout.cells[s].y + self.layout.cells[s].h
        } else { 0 }
    }

    fn max_scroll(&self) -> usize {
        self.content_h().saturating_sub(self.viewport_h())
    }

    fn ensure_sel_visible(&mut self) {
        if self.layout.cells.is_empty() { return; }
        let cell = &self.layout.cells[self.sel];
        let vp = self.viewport_h();
        if cell.y < self.scroll_y {
            self.scroll_y = cell.y;
        }
        if cell.y + cell.h > self.scroll_y + vp {
            self.scroll_y = (cell.y + cell.h).saturating_sub(vp);
        }
        self.scroll_y = self.scroll_y.min(self.max_scroll());
    }

    /// Returns (start_row, end_row) of rows intersecting the viewport.
    fn visible_row_range(&self) -> (usize, usize) {
        let top = self.scroll_y;
        let bottom = self.scroll_y + self.viewport_h();
        let mut start = 0;
        let mut end = self.layout.row_ranges.len();
        for (i, &(s, _)) in self.layout.row_ranges.iter().enumerate() {
            let row_top = self.layout.cells[s].y;
            let row_bottom = row_top + self.layout.cells[s].h;
            if row_bottom <= top { start = i + 1; }
            if row_top >= bottom && end == self.layout.row_ranges.len() { end = i; }
        }
        (start, end)
    }

    fn render(&mut self) {
        match self.state {
            ViewState::Grid => self.render_grid(),
            ViewState::Full => self.render_full(),
        }
    }

    fn render_grid(&mut self) {
        let (w, h) = (self.win_w, self.win_h);
        self.buf.fill(BG);

        if self.layout.row_ranges.is_empty() { return; }

        let (start_row, end_row) = self.visible_row_range();

        for row_idx in start_row..end_row {
            let (item_start, item_end) = self.layout.row_ranges[row_idx];
            for i in item_start..item_end {
                let cell = &self.layout.cells[i];
                let sy = cell.y as isize - self.scroll_y as isize;

                if let Some(ref thumb) = self.items[i].thumb {
                    blit(&mut self.buf, w, h, thumb, cell.x, sy);
                } else {
                    draw_placeholder(&mut self.buf, w, h, cell.x, sy, cell.w, cell.h, self.frame);
                }

                if i == self.sel {
                    draw_selection(&mut self.buf, w, h, cell.x, sy, cell.w, cell.h, SEL_COLOR);
                }
            }
        }

        let item = &self.items[self.sel];
        let name = std::path::Path::new(&item.path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| item.path.clone());
        let status = if item.score > 0.0 {
            format!(" {} | {:.2} | {}/{} | Arrows:navigate  Enter:view  q:quit",
                name, item.score, self.sel + 1, self.items.len())
        } else {
            format!(" {} | {}/{} | Arrows:navigate  Enter:view  q:quit",
                name, self.sel + 1, self.items.len())
        };
        draw_bar(&mut self.buf, w, h, &status);
    }

    fn render_full(&mut self) {
        let (w, h) = (self.win_w, self.win_h);
        self.buf.fill(0x00000000);

        let usable_h = h.saturating_sub(BAR_HEIGHT).max(1);

        if let Some(ref full) = self.items[self.sel].full {
            let tx = w.saturating_sub(full.w) / 2;
            let ty = usable_h.saturating_sub(full.h) / 2;
            blit(&mut self.buf, w, h, full, tx, ty as isize);
        } else {
            let pw = 300.min(w);
            let ph = 60.min(usable_h);
            let px = w.saturating_sub(pw) / 2;
            let py = usable_h.saturating_sub(ph) / 2;
            draw_placeholder(&mut self.buf, w, h, px, py as isize, pw, ph, self.frame);
        }

        let name = std::path::Path::new(&self.items[self.sel].path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| self.items[self.sel].path.clone());
        let status = if self.items[self.sel].score > 0.0 {
            format!(" {} | {:.2} | {}/{} | Left/Right:prev/next  Esc:back  q:quit",
                name, self.items[self.sel].score, self.sel + 1, self.items.len())
        } else {
            format!(" {} | {}/{} | Left/Right:prev/next  Esc:back  q:quit",
                name, self.sel + 1, self.items.len())
        };
        draw_bar(&mut self.buf, w, h, &status);
    }

    fn nav_up(&self) -> usize {
        if self.layout.item_to_row.is_empty() { return self.sel; }
        let row = self.layout.item_to_row[self.sel];
        if row == 0 { return self.sel; }
        let cx = self.layout.cells[self.sel].x + self.layout.cells[self.sel].w / 2;
        let (s, e) = self.layout.row_ranges[row - 1];
        (s..e).min_by_key(|&j| {
            let jx = self.layout.cells[j].x + self.layout.cells[j].w / 2;
            (cx as isize - jx as isize).unsigned_abs()
        }).unwrap_or(self.sel)
    }

    fn nav_down(&self) -> usize {
        if self.layout.item_to_row.is_empty() { return self.sel; }
        let row = self.layout.item_to_row[self.sel];
        if row + 1 >= self.layout.row_ranges.len() { return self.sel; }
        let cx = self.layout.cells[self.sel].x + self.layout.cells[self.sel].w / 2;
        let (s, e) = self.layout.row_ranges[row + 1];
        (s..e).min_by_key(|&j| {
            let jx = self.layout.cells[j].x + self.layout.cells[j].w / 2;
            (cx as isize - jx as isize).unsigned_abs()
        }).unwrap_or(self.sel)
    }

    /// Load one unloaded item near the viewport. Returns true if something was loaded.
    fn load_next_visible(&mut self) -> bool {
        if self.layout.row_ranges.is_empty() { return false; }
        let (start_row, end_row) = self.visible_row_range();
        // Extend range for prefetch
        let start_row = start_row.saturating_sub(1);
        let end_row = (end_row + PREFETCH_ROWS).min(self.layout.row_ranges.len());
        for row_idx in start_row..end_row {
            let (item_start, item_end) = self.layout.row_ranges[row_idx];
            for i in item_start..item_end {
                if self.items[i].thumb.is_none() {
                    self.items[i].ensure_img();
                    let cell = &self.layout.cells[i];
                    let t = render_fill(
                        self.items[i].img.as_ref().unwrap(), cell.w, cell.h,
                    );
                    self.items[i].thumb = Some(t);
                    self.dirty = true;
                    return true;
                }
            }
        }
        false
    }

    /// Load fullscreen image if needed. Returns true if something was loaded.
    fn load_full_if_needed(&mut self) -> bool {
        if self.items[self.sel].full.is_none() {
            self.items[self.sel].ensure_img();
            let usable_h = self.win_h.saturating_sub(BAR_HEIGHT).max(1);
            let t = render_fit(self.items[self.sel].img.as_ref().unwrap(), self.win_w, usable_h);
            self.items[self.sel].full = Some(t);
            self.dirty = true;

            // Pre-cache adjacent
            let total = self.items.len();
            let (w, usable) = (self.win_w, usable_h);
            for &adj in &[self.sel.wrapping_sub(1), self.sel + 1] {
                if adj < total && self.items[adj].full.is_none() {
                    self.items[adj].ensure_img();
                    let t = render_fit(self.items[adj].img.as_ref().unwrap(), w, usable);
                    self.items[adj].full = Some(t);
                }
            }
            return true;
        }
        false
    }

    /// Find which item is at pixel (x, y) in the grid, accounting for scroll.
    fn hit_test(&self, x: usize, y: usize) -> Option<usize> {
        let abs_y = y + self.scroll_y;
        for (i, cell) in self.layout.cells.iter().enumerate() {
            if x >= cell.x && x < cell.x + cell.w
                && abs_y >= cell.y && abs_y < cell.y + cell.h {
                return Some(i);
            }
        }
        None
    }

    fn handle_input(&mut self, window: &Window) {
        let total = self.items.len();
        let old_sel = self.sel;

        // Mouse scroll (both grid and fullscreen)
        if let Some((_, scroll_y)) = window.get_scroll_wheel() {
            match self.state {
                ViewState::Grid => {
                    let step = 60;
                    if scroll_y > 0.0 {
                        self.scroll_y = self.scroll_y.saturating_sub(step);
                    } else if scroll_y < 0.0 {
                        self.scroll_y = (self.scroll_y + step).min(self.max_scroll());
                    }
                    self.dirty = true;
                }
                ViewState::Full => {
                    if scroll_y > 0.0 {
                        self.sel = self.sel.saturating_sub(1);
                    } else if scroll_y < 0.0 {
                        self.sel = (self.sel + 1).min(total - 1);
                    }
                }
            }
        }

        // Mouse click
        if window.get_mouse_down(MouseButton::Left) {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
                let (mx, my) = (mx as usize, my as usize);
                match self.state {
                    ViewState::Grid => {
                        if my < self.viewport_h() {
                            if let Some(idx) = self.hit_test(mx, my) {
                                if self.sel == idx {
                                    // Double-click effect: already selected → fullscreen
                                    self.state = ViewState::Full;
                                }
                                self.sel = idx;
                                self.dirty = true;
                            }
                        }
                    }
                    ViewState::Full => {}
                }
            }
        }

        match self.state {
            ViewState::Grid => {
                if window.is_key_pressed(Key::Escape, KeyRepeat::No)
                    || window.is_key_pressed(Key::Q, KeyRepeat::No) {
                    self.quit = true; return;
                }
                if window.is_key_pressed(Key::Enter, KeyRepeat::No) {
                    self.state = ViewState::Full;
                    self.dirty = true; return;
                }
                if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                    self.sel = (self.sel + 1).min(total - 1);
                }
                if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
                    self.sel = self.sel.saturating_sub(1);
                }
                if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
                    self.sel = self.nav_down();
                }
                if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
                    self.sel = self.nav_up();
                }
                if window.is_key_pressed(Key::PageDown, KeyRepeat::Yes) {
                    let vp = self.viewport_h();
                    self.scroll_y = (self.scroll_y + vp).min(self.max_scroll());
                    // Move selection into view
                    let (start_row, _) = self.visible_row_range();
                    if start_row < self.layout.row_ranges.len() {
                        self.sel = self.layout.row_ranges[start_row].0;
                    }
                }
                if window.is_key_pressed(Key::PageUp, KeyRepeat::Yes) {
                    let vp = self.viewport_h();
                    self.scroll_y = self.scroll_y.saturating_sub(vp);
                    let (start_row, _) = self.visible_row_range();
                    if start_row < self.layout.row_ranges.len() {
                        self.sel = self.layout.row_ranges[start_row].0;
                    }
                }
                if window.is_key_pressed(Key::Home, KeyRepeat::No) { self.sel = 0; }
                if window.is_key_pressed(Key::End, KeyRepeat::No) { self.sel = total - 1; }
            }
            ViewState::Full => {
                if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
                    self.state = ViewState::Grid;
                    self.dirty = true; return;
                }
                if window.is_key_pressed(Key::Q, KeyRepeat::No) {
                    self.quit = true; return;
                }
                if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                    self.sel = (self.sel + 1).min(total - 1);
                }
                if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
                    self.sel = self.sel.saturating_sub(1);
                }
            }
        }

        if self.sel != old_sel {
            self.ensure_sel_visible();
            self.dirty = true;
        }
    }
}

// ── Suppress Wayland cleanup warnings ───────────────────────────────────────

fn suppress_stderr<F: FnOnce()>(f: F) {
    unsafe {
        let saved = libc::dup(2);
        let devnull = libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
        if devnull >= 0 { libc::dup2(devnull, 2); libc::close(devnull); }
        f();
        if saved >= 0 { libc::dup2(saved, 2); libc::close(saved); }
    }
}

// ── Public entry point ──────────────────────────────────────────────────────

pub fn run(results: &[(f64, String)]) -> Result<()> {
    if results.is_empty() { return Ok(()); }

    let (init_w, init_h) = (1280, 720);
    let window = Window::new("nanoimg", init_w, init_h,
        WindowOptions { resize: true, ..Default::default() });
    let mut window = match window {
        Ok(w) => w,
        Err(e) => { eprintln!("nanoimg: could not open viewer: {}", e); return Ok(()); }
    };
    window.set_target_fps(60);

    let mut viewer = Viewer::new(results, init_w, init_h);

    while window.is_open() && !viewer.quit {
        let (w, h) = window.get_size();
        if w > 0 && h > 0 && (w != viewer.win_w || h != viewer.win_h) {
            viewer.resize(w, h);
        }

        viewer.handle_input(&window);

        // Progressive loading: load one item per frame to stay responsive
        match viewer.state {
            ViewState::Grid => { viewer.load_next_visible(); }
            ViewState::Full => { viewer.load_full_if_needed(); }
        }

        // Animate loading placeholders
        viewer.frame = viewer.frame.wrapping_add(1);
        if viewer.frame % 15 == 0 {
            let has_loading = match viewer.state {
                ViewState::Grid => {
                    let (sr, er) = viewer.visible_row_range();
                    (sr..er).any(|ri| {
                        let (s, e) = viewer.layout.row_ranges[ri];
                        (s..e).any(|i| viewer.items[i].thumb.is_none())
                    })
                }
                ViewState::Full => viewer.items[viewer.sel].full.is_none(),
            };
            if has_loading { viewer.dirty = true; }
        }

        if viewer.dirty {
            viewer.render();
            viewer.dirty = false;
        }
        window.update_with_buffer(&viewer.buf, viewer.win_w, viewer.win_h)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    suppress_stderr(move || drop(window));
    Ok(())
}
