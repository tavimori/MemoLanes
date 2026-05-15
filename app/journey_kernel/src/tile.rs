use crate::bitmap2d::{bitvec_from_bytes_lsb, bitvec_to_bytes_lsb, BitMap2D};
use crate::tile_iter::{MipmapIter, OverscanIter, SubtileIter, TilePixelIter};
use bitvec::prelude::*;

#[derive(Clone)]
pub struct GenericTile {
    sub_tiles: Vec<Option<Box<GenericTile>>>,
    bitmap: Option<BitMap2D>,
    width_exp: i16,
    is_leaf: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingSubTile {
    pub x: i64,
    pub y: i64,
    pub parent_width_exp: i16,
}

pub fn xy_to_index(x: i64, y: i64, width_exp: i16) -> usize {
    (x + y * (1 << width_exp)) as usize
}

pub fn index_to_xy(index: usize, width_exp: i16) -> (i64, i64) {
    (
        index as i64 % (1 << width_exp),
        index as i64 / (1 << width_exp),
    )
}


impl GenericTile {
    pub fn new(width_exp: i16) -> Self {
        let sub_tiles = vec![None; 1 << (width_exp * 2)];

        Self {
            sub_tiles,
            bitmap: None,
            width_exp,
            is_leaf: false,
        }
    }

    pub fn set_sub_tile(&mut self, x: i64, y: i64, tile: GenericTile) {
        self.sub_tiles[xy_to_index(x, y, self.width_exp)] = Some(Box::new(tile));
    }

    pub fn width_exp(&self) -> i16 {
        self.width_exp
    }

    pub fn mipmap_levels(&self) -> Vec<BitVec> {
        match &self.bitmap {
            Some(bm) => (0..bm.num_levels())
                .map(|k| bm.level_at_offset(k).unwrap().clone())
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn set_mipmap(&mut self, mipmap: Vec<BitVec>) {
        if mipmap.is_empty() {
            self.bitmap = None;
            return;
        }
        let base = mipmap[0].clone();
        let width_exp = (base.len() as f64).sqrt().log2() as u8;
        let lods = mipmap[1..].to_vec();
        self.bitmap = Some(BitMap2D::from_precomputed(width_exp, base, lods));
    }

    pub fn has_sub_tile(&self, x: i64, y: i64) -> bool {
        self.sub_tiles[xy_to_index(x, y, self.width_exp)].is_some()
    }

    pub fn get_sub_tile(&self, x: i64, y: i64) -> Option<&GenericTile> {
        self.sub_tiles[xy_to_index(x, y, self.width_exp)].as_deref()
    }

    pub fn evict_sub_tile(&mut self, x: i64, y: i64) {
        let idx = xy_to_index(x, y, self.width_exp);
        self.sub_tiles[idx] = None;
    }

    pub fn downgrade_sub_tile_to_mipmap(&mut self, x: i64, y: i64) {
        let idx = xy_to_index(x, y, self.width_exp);
        if let Some(tile) = self.sub_tiles[idx].as_mut() {
            tile.sub_tiles = vec![];
            tile.is_leaf = true;
        }
    }

    fn mipmap_has_subtile(&self, idx: usize) -> bool {
        self.bitmap
            .as_ref()
            .map(|bm| {
                let bits = bm.as_bitvec();
                idx < bits.len() && bits[idx]
            })
            .unwrap_or(false)
    }

    pub fn check_ready(&self, x: i64, y: i64, z: i16, resolution_exp: i16) -> Vec<MissingSubTile> {
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);
        debug_assert!(z >= 0);

        if z + resolution_exp <= self.width_exp {
            // Mipmap is sufficient for this query resolution.
            return Vec::new();
        }

        if z <= self.width_exp {
            if self.is_leaf {
                return Vec::new();
            }

            let mut missing = Vec::new();
            let subtile_resolution_exp = z + resolution_exp - self.width_exp;
            let span_exp = self.width_exp - z;
            let x_min = x << span_exp;
            let x_max = (x + 1) << span_exp;
            let y_min = y << span_exp;
            let y_max = (y + 1) << span_exp;

            for sy in y_min..y_max {
                for sx in x_min..x_max {
                    let idx = xy_to_index(sx, sy, self.width_exp);
                    if let Some(sub_tile) = &self.sub_tiles[idx] {
                        missing.extend(sub_tile.check_ready(0, 0, 0, subtile_resolution_exp));
                    } else if self.mipmap_has_subtile(idx) {
                        missing.push(MissingSubTile {
                            x: sx,
                            y: sy,
                            parent_width_exp: self.width_exp,
                        });
                    }
                }
            }

            return missing;
        }

        if self.is_leaf {
            return Vec::new();
        }

        let child_z = z - self.width_exp;
        let tile_x = x >> child_z;
        let tile_y = y >> child_z;
        let child_x = x - (tile_x << child_z);
        let child_y = y - (tile_y << child_z);
        let idx = xy_to_index(tile_x, tile_y, self.width_exp);

        if let Some(sub_tile) = &self.sub_tiles[idx] {
            return sub_tile.check_ready(child_x, child_y, child_z, resolution_exp);
        }

        if self.mipmap_has_subtile(idx) {
            return vec![MissingSubTile {
                x: tile_x,
                y: tile_y,
                parent_width_exp: self.width_exp,
            }];
        }

        Vec::new()
    }

    pub fn iter_present_sub_tiles(&self) -> impl Iterator<Item = (i64, i64, &GenericTile)> {
        self.sub_tiles
            .iter()
            .enumerate()
            .filter_map(move |(idx, child)| {
                child.as_deref().map(|tile| {
                    let (x, y) = index_to_xy(idx, self.width_exp);
                    (x, y, tile)
                })
            })
    }

    /// Construct a leaf tile from an MSB-ordered packed bitmap payload.
    ///
    /// This is used by legacy Fog of World/JBM bitmap payloads where bit
    /// index 0 maps to the most-significant bit of byte 0.
    pub fn from_msb_bitmap(width_exp: i16, data: &[u8]) -> Self {
        let width: usize = 1 << width_exp;
        let expected_bytes = (width * width) / 8;
        assert_eq!(data.len(), expected_bytes, "Data length mismatch");

        let mut bits = BitVec::with_capacity(width * width);
        bits.resize(width * width, false);

        for (byte_idx, byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                let pixel_idx = byte_idx * 8 + bit_idx;
                bits.set(pixel_idx, ((*byte >> (7 - bit_idx)) & 1) == 1);
            }
        }

        let mut bm = BitMap2D::from_bitvec(width_exp as u8, bits);
        bm.build_lods();

        Self {
            sub_tiles: vec![],
            bitmap: Some(bm),
            width_exp,
            is_leaf: true,
        }
    }

    /// Construct a leaf tile from precomputed mipmap levels.
    pub fn from_precomputed_mipmap(width_exp: i16, mipmap: Vec<BitVec>) -> Self {
        let bitmap = if mipmap.is_empty() {
            None
        } else {
            let base = mipmap[0].clone();
            let lods = mipmap[1..].to_vec();
            Some(BitMap2D::from_precomputed(width_exp as u8, base, lods))
        };
        Self {
            sub_tiles: vec![],
            bitmap,
            width_exp,
            is_leaf: true,
        }
    }

    /// Extract a tile's mipmap pyramid for `(x, y, z, resolution_exp)` in LSB order.
    ///
    /// Returns `None` when the extracted tile is empty.
    pub fn extract_mipmap(
        &self,
        x: i64,
        y: i64,
        z: i16,
        resolution_exp: i16,
    ) -> Option<Vec<BitVec>> {
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);
        debug_assert!(z >= 0);
        debug_assert!(resolution_exp >= 0);

        // Case 1: Navigate tree for deep zoom
        if z > self.width_exp {
            if self.is_leaf {
                return None;
            }
            let child_z = z - self.width_exp;
            let tile_x = x >> child_z;
            let tile_y = y >> child_z;
            let child_x = x - (tile_x << child_z);
            let child_y = y - (tile_y << child_z);
            let idx = xy_to_index(tile_x, tile_y, self.width_exp);
            return self.sub_tiles[idx].as_deref().and_then(|sub_tile| {
                sub_tile.extract_mipmap(child_x, child_y, child_z, resolution_exp)
            });
        }

        // Case 2: Leaf tile — reuse existing mipmap levels directly
        if self.is_leaf && z == 0 {
            if resolution_exp <= self.width_exp {
                let bm = match &self.bitmap {
                    Some(bm) => bm,
                    None => return None,
                };
                let level_offset = (self.width_exp - resolution_exp) as usize;
                if level_offset >= bm.num_levels() {
                    return None;
                }
                if bm.level_at_offset(level_offset).unwrap().not_any() {
                    return None;
                }
                return Some(
                    (level_offset..bm.num_levels())
                        .map(|k| bm.level_at_offset(k).unwrap().clone())
                        .collect(),
                );
            }
            // resolution_exp > width_exp: oversample leaf data
            return self.oversample_leaf_mipmap(resolution_exp);
        }

        // Case 3: Non-leaf tile, z <= width_exp — composite sub-tile mipmaps
        if !self.is_leaf {
            let span_exp = (self.width_exp - z) as usize;
            let sub_res_exp = resolution_exp as isize - span_exp as isize;
            if sub_res_exp >= 0 {
                return self.composite_sub_tile_mipmaps(
                    x,
                    y,
                    span_exp,
                    sub_res_exp as i16,
                    resolution_exp,
                );
            }
        }

        // Fallback: iter_pixels for rare edge cases
        let side = 1usize << resolution_exp;
        let mut level0 = BitVec::repeat(false, side * side);
        for (px, py) in self.iter_pixels(0, 0, x, y, z, resolution_exp) {
            level0.set(xy_to_index(px, py, resolution_exp), true);
        }
        if level0.not_any() {
            return None;
        }
        Some(Self::build_leaf_mipmap_from_level0(resolution_exp, level0))
    }

    /// Build a mipmap for a leaf tile at a resolution finer than its native data.
    /// Each native pixel is expanded to a `2^overscan × 2^overscan` block.
    fn oversample_leaf_mipmap(&self, resolution_exp: i16) -> Option<Vec<BitVec>> {
        let bm = self.bitmap.as_ref()?;
        let src_level0 = bm.as_bitvec();
        if src_level0.not_any() {
            return None;
        }

        let overscan = (resolution_exp - self.width_exp) as usize;
        let src_side = 1usize << self.width_exp;
        let mut result = Vec::with_capacity(resolution_exp as usize + 1);

        for k in 0..overscan {
            let pixel_repeat = 1usize << (overscan - k);
            let out_side = 1usize << (resolution_exp as usize - k);
            let bits_per_word = usize::BITS as usize;
            let word_count = (out_side * out_side).div_ceil(bits_per_word);
            let mut words = vec![0usize; word_count];

            for src_y in 0..src_side {
                for src_x in 0..src_side {
                    if src_level0[src_y * src_side + src_x] {
                        for dy in 0..pixel_repeat {
                            let out_y = src_y * pixel_repeat + dy;
                            for dx in 0..pixel_repeat {
                                let out_x = src_x * pixel_repeat + dx;
                                let bit_pos = out_y * out_side + out_x;
                                words[bit_pos / bits_per_word] |=
                                    1usize << (bit_pos % bits_per_word);
                            }
                        }
                    }
                }
            }

            let mut bv = BitVec::from_vec(words);
            bv.truncate(out_side * out_side);
            result.push(bv);
        }

        for k in 0..bm.num_levels() {
            result.push(bm.level_at_offset(k).unwrap().clone());
        }

        Some(result)
    }

    /// Composite sub-tile mipmaps into a single output mipmap using word-level
    /// bulk copies where alignment permits.
    fn composite_sub_tile_mipmaps(
        &self,
        x: i64,
        y: i64,
        span_exp: usize,
        sub_res_exp: i16,
        resolution_exp: i16,
    ) -> Option<Vec<BitVec>> {
        let grid_side = 1usize << span_exp;
        let x_base = x * (1i64 << span_exp);
        let y_base = y * (1i64 << span_exp);
        let sub_side = 1usize << sub_res_exp;
        let out_side = 1usize << resolution_exp;
        let num_sub_levels = sub_res_exp as usize + 1;
        let bits_per_word = usize::BITS as usize;

        let mut out_word_vecs: Vec<Vec<usize>> = (0..num_sub_levels)
            .map(|k| {
                let level_side = out_side >> k;
                let total_bits = level_side * level_side;
                vec![0usize; total_bits.div_ceil(bits_per_word)]
            })
            .collect();

        let mut has_any = false;

        for gy in 0..grid_side {
            for gx in 0..grid_side {
                let sx = x_base + gx as i64;
                let sy = y_base + gy as i64;
                let idx = xy_to_index(sx, sy, self.width_exp);
                let sub_tile = match &self.sub_tiles[idx] {
                    Some(t) => t,
                    None => continue,
                };
                let sub_mipmap = match sub_tile.extract_mipmap(0, 0, 0, sub_res_exp) {
                    Some(m) => m,
                    None => continue,
                };
                has_any = true;

                for k in 0..num_sub_levels.min(sub_mipmap.len()) {
                    let sub_level = &sub_mipmap[k];
                    let sub_level_side = sub_side >> k;
                    let out_level_side = out_side >> k;
                    let out_words = &mut out_word_vecs[k];

                    if sub_level_side >= bits_per_word
                        && sub_level_side.is_multiple_of(bits_per_word)
                    {
                        let src_words = sub_level.as_raw_slice();
                        let src_words_per_row = sub_level_side / bits_per_word;
                        let out_words_per_row = out_level_side / bits_per_word;
                        for row in 0..sub_level_side {
                            let dst_row = gy * sub_level_side + row;
                            let dst_word_col = gx * sub_level_side / bits_per_word;
                            let src_row_start = row * src_words_per_row;
                            let dst_row_start = dst_row * out_words_per_row + dst_word_col;
                            for w in 0..src_words_per_row {
                                out_words[dst_row_start + w] |= src_words[src_row_start + w];
                            }
                        }
                    } else {
                        for row in 0..sub_level_side {
                            let src_start = row * sub_level_side;
                            let dst_row = gy * sub_level_side + row;
                            let dst_start = dst_row * out_level_side + gx * sub_level_side;
                            for bit in 0..sub_level_side {
                                if sub_level[src_start + bit] {
                                    let dst_bit = dst_start + bit;
                                    out_words[dst_bit / bits_per_word] |=
                                        1usize << (dst_bit % bits_per_word);
                                }
                            }
                        }
                    }
                }
            }
        }

        if !has_any {
            return None;
        }

        let mut out_mipmap: Vec<BitVec> = out_word_vecs
            .into_iter()
            .enumerate()
            .map(|(k, words)| {
                let level_side = out_side >> k;
                let mut bv = BitVec::from_vec(words);
                bv.truncate(level_side * level_side);
                bv
            })
            .collect();

        // Build remaining pyramid levels beyond the composited ones
        let last_level = out_mipmap.last().unwrap().clone();
        let last_exp = (out_side >> (num_sub_levels - 1)).trailing_zeros() as u8;
        if last_exp > 0 {
            let mut tail_bm = BitMap2D::from_bitvec(last_exp, last_level);
            tail_bm.build_lods();
            for lod in tail_bm.lod_levels() {
                out_mipmap.push(lod.clone());
            }
        }

        Some(out_mipmap)
    }

    /// Iterate pixels over the bitmap tile at a given resolution.
    ///
    /// If a subtile slot is `None`, it is treated as absent and skipped.
    /// Call [`GenericTile::check_ready`] before iterating when full-precision
    /// results are required from partially loaded trees.
    pub fn iter_pixels(
        &self,
        start_x: i64,
        start_y: i64,
        x: i64,
        y: i64,
        z: i16,
        resolution_exp: i16,
    ) -> TilePixelIter<'_> {
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);
        debug_assert!(z >= 0);

        if z > 0 {
            debug_assert_eq!(start_x, 0);
            debug_assert_eq!(start_y, 0);
        }

        if z + resolution_exp <= self.width_exp {
            let buffer_exp = z + resolution_exp;
            let level_offset = (self.width_exp - resolution_exp - z) as usize;
            let buffer = self.bitmap.as_ref().unwrap().level_at_offset(level_offset).unwrap();

            debug_assert_eq!(buffer.len(), 1 << (2 * (z + resolution_exp)));

            TilePixelIter::MipmapIter(MipmapIter::new(
                buffer, start_x, start_y, x, y, z, buffer_exp,
            ))
        } else if z <= self.width_exp {
            if self.is_leaf {
                TilePixelIter::OverscanIter(OverscanIter::new(
                    self.bitmap.as_ref().unwrap().as_bitvec(),
                    start_x,
                    start_y,
                    x,
                    y,
                    z,
                    self.width_exp,
                    z + resolution_exp - self.width_exp,
                ))
            } else {
                TilePixelIter::SubTileIter(SubtileIter::new(
                    &self.sub_tiles,
                    start_x,
                    start_y,
                    x,
                    y,
                    z,
                    self.width_exp,
                    z + resolution_exp - self.width_exp,
                ))
            }
        } else {
            let z = z - self.width_exp;

            let tile_x = x >> z;
            let tile_y = y >> z;
            let x = x - (tile_x << z);
            let y = y - (tile_y << z);

            // TODO: make sure if start_x and start_y = 0

            if self.is_leaf {
                // TODO: check if this is correct
                unimplemented!();
            } else if let Some(sub_tile) =
                &self.sub_tiles[xy_to_index(tile_x, tile_y, self.width_exp)]
            {
                sub_tile.iter_pixels(start_x, start_y, x, y, z, resolution_exp)
            } else {
                // bitmap is empty
                TilePixelIter::Empty
            }
        }
    }

    pub fn compute_mipmap(&mut self) {
        let mut bm = BitMap2D::new(self.width_exp as u8);
        let width = 1usize << self.width_exp;
        for i in 0..width {
            for j in 0..width {
                let idx = i * width + j;
                if self.sub_tiles[idx].is_some() {
                    bm.set(j, i, true);
                }
            }
        }
        bm.build_lods();
        self.bitmap = Some(bm);
    }

    pub(crate) fn build_leaf_mipmap_from_level0(width_exp: i16, level0: BitVec) -> Vec<BitVec> {
        let mut bm = BitMap2D::from_bitvec(width_exp as u8, level0);
        bm.build_lods();
        bm.into_all_levels()
    }

    /// Serialize the tile to bytes
    /// Format:
    /// - width_exp: i16 (2 bytes)
    /// - is_leaf: u8 (1 byte, 0 or 1)
    /// - mipmap_count: u32 (4 bytes)
    /// - for each mipmap level:
    ///   - bit_count: u64 (8 bytes)
    ///   - bytes: [u8] (variable length, aligned to bytes)
    /// - sub_tiles_count: u32 (4 bytes)
    /// - for each sub_tile:
    ///   - exists: u8 (1 byte, 0 or 1)
    ///   - if exists: recursively serialized tile data
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.width_exp.to_le_bytes());
        bytes.push(if self.is_leaf { 1 } else { 0 });

        let num_levels = self.bitmap.as_ref().map_or(0, |bm| bm.num_levels());
        bytes.extend_from_slice(&(num_levels as u32).to_le_bytes());
        if let Some(bm) = &self.bitmap {
            for k in 0..bm.num_levels() {
                let level = bm.level_at_offset(k).unwrap();
                let bit_count = level.len() as u64;
                bytes.extend_from_slice(&bit_count.to_le_bytes());
                bytes.extend_from_slice(&bitvec_to_bytes_lsb(level));
            }
        }

        bytes.extend_from_slice(&(self.sub_tiles.len() as u32).to_le_bytes());
        for sub_tile_opt in &self.sub_tiles {
            if let Some(sub_tile) = sub_tile_opt {
                bytes.push(1);
                let sub_tile_bytes = sub_tile.to_bytes();
                bytes.extend_from_slice(&(sub_tile_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&sub_tile_bytes);
            } else {
                bytes.push(0);
            }
        }

        bytes
    }

    /// Deserialize a tile from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut offset = 0;

        if bytes.len() < offset + 2 {
            return Err("Insufficient data for width_exp".to_string());
        }
        let width_exp = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        offset += 2;

        if bytes.len() < offset + 1 {
            return Err("Insufficient data for is_leaf".to_string());
        }
        let is_leaf = bytes[offset] != 0;
        offset += 1;

        if bytes.len() < offset + 4 {
            return Err("Insufficient data for mipmap_count".to_string());
        }
        let mipmap_count = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut levels = Vec::with_capacity(mipmap_count);
        for _ in 0..mipmap_count {
            if bytes.len() < offset + 8 {
                return Err("Insufficient data for bit_count".to_string());
            }
            let bit_count = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]) as usize;
            offset += 8;

            let byte_count = bit_count.div_ceil(8);
            if bytes.len() < offset + byte_count {
                return Err("Insufficient data for mipmap level".to_string());
            }

            levels.push(bitvec_from_bytes_lsb(
                &bytes[offset..offset + byte_count],
                bit_count,
            ));
            offset += byte_count;
        }

        if bytes.len() < offset + 4 {
            return Err("Insufficient data for sub_tiles_count".to_string());
        }
        let sub_tiles_count = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut sub_tiles = Vec::with_capacity(sub_tiles_count);
        for _ in 0..sub_tiles_count {
            if bytes.len() < offset + 1 {
                return Err("Insufficient data for sub_tile exists flag".to_string());
            }
            let exists = bytes[offset] != 0;
            offset += 1;

            if exists {
                if bytes.len() < offset + 4 {
                    return Err("Insufficient data for sub_tile length".to_string());
                }
                let sub_tile_len = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize;
                offset += 4;

                if bytes.len() < offset + sub_tile_len {
                    return Err("Insufficient data for sub_tile data".to_string());
                }
                let sub_tile = Self::from_bytes(&bytes[offset..offset + sub_tile_len])?;
                sub_tiles.push(Some(Box::new(sub_tile)));
                offset += sub_tile_len;
            } else {
                sub_tiles.push(None);
            }
        }

        let bitmap = if levels.is_empty() {
            None
        } else {
            let base = levels.remove(0);
            Some(BitMap2D::from_precomputed(width_exp as u8, base, levels))
        };

        Ok(Self {
            sub_tiles,
            bitmap,
            width_exp,
            is_leaf,
        })
    }
}

pub fn get_tile_pixels(
    tile: &GenericTile,
    x: i64,
    y: i64,
    z: i16,
    resolution_exp: i16,
) -> Vec<(i64, i64)> {
    let iter = tile.iter_pixels(0, 0, x, y, z, resolution_exp);
    iter.collect()
}
