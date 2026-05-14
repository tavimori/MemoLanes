use crate::tile::xy_to_index;

pub fn encode_tile_bitmap(pixels: &[(i64, i64)], tile_bitmap_exp: u8) -> Result<Vec<u8>, String> {
    let side = 1i64 << tile_bitmap_exp;
    let mut bitmap = vec![0u8; bitmap_bytes_for_exp(tile_bitmap_exp)?];
    for &(x, y) in pixels {
        if x < 0 || y < 0 || x >= side || y >= side {
            return Err(format!(
                "pixel ({}, {}) is outside bitmap bounds [0, {})",
                x, y, side
            ));
        }
        let idx = xy_to_index(x, y, tile_bitmap_exp as i16);
        set_lsb_bit(&mut bitmap, idx, true);
    }
    Ok(bitmap)
}

pub fn decode_tile_bitmap(bitmap: &[u8], tile_bitmap_exp: u8) -> Result<Vec<(i64, i64)>, String> {
    let expected = bitmap_bytes_for_exp(tile_bitmap_exp)?;
    if bitmap.len() != expected {
        return Err(format!(
            "bitmap length mismatch: expected {}, got {}",
            expected,
            bitmap.len()
        ));
    }

    let side = 1usize << tile_bitmap_exp;
    let pixel_count = side * side;
    let mut out = Vec::new();
    for idx in 0..pixel_count {
        if test_lsb_bit(bitmap, idx) {
            let x = (idx % side) as i64;
            let y = (idx / side) as i64;
            out.push((x, y));
        }
    }
    Ok(out)
}

pub fn encode_tile_coord_list(
    pixels: &[(i64, i64)],
    tile_bitmap_exp: u8,
) -> Result<Vec<u8>, String> {
    let side = 1i64 << tile_bitmap_exp;
    let mut out = Vec::with_capacity(pixels.len() * 4);

    let mut sorted = pixels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    for (x, y) in sorted {
        if x < 0 || y < 0 || x >= side || y >= side {
            return Err(format!(
                "pixel ({}, {}) is outside bitmap bounds [0, {})",
                x, y, side
            ));
        }
        out.extend_from_slice(&(x as u16).to_le_bytes());
        out.extend_from_slice(&(y as u16).to_le_bytes());
    }

    Ok(out)
}

pub fn decode_tile_coord_list(
    bytes: &[u8],
    tile_bitmap_exp: u8,
) -> Result<Vec<(i64, i64)>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("coordinate list byte length must be divisible by 4".to_string());
    }

    let side = 1u16 << tile_bitmap_exp;
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let x = u16::from_le_bytes([chunk[0], chunk[1]]);
        let y = u16::from_le_bytes([chunk[2], chunk[3]]);
        if x >= side || y >= side {
            return Err(format!(
                "decoded coordinate ({}, {}) is outside bitmap bounds [0, {})",
                x, y, side
            ));
        }
        out.push((x as i64, y as i64));
    }
    Ok(out)
}

/// Sorted flat u32 indices: each pixel packed as `x + y * side`.
/// For exp=10 each index uses only 20 of 32 bits; the sorted sequence
/// compresses dramatically better than separate u16 pairs.
pub fn encode_tile_coord_u32(
    pixels: &[(i64, i64)],
    tile_bitmap_exp: u8,
) -> Result<Vec<u8>, String> {
    let side = 1i64 << tile_bitmap_exp;
    let mut indices: Vec<u32> = Vec::with_capacity(pixels.len());

    for &(x, y) in pixels {
        if x < 0 || y < 0 || x >= side || y >= side {
            return Err(format!(
                "pixel ({}, {}) is outside bitmap bounds [0, {})",
                x, y, side
            ));
        }
        indices.push((x + y * side) as u32);
    }
    indices.sort_unstable();
    indices.dedup();

    let mut out = Vec::with_capacity(indices.len() * 4);
    for idx in indices {
        out.extend_from_slice(&idx.to_le_bytes());
    }
    Ok(out)
}

pub fn decode_tile_coord_u32(bytes: &[u8], tile_bitmap_exp: u8) -> Result<Vec<(i64, i64)>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("u32 index list byte length must be divisible by 4".to_string());
    }

    let side = 1u32 << tile_bitmap_exp;
    let max_idx = side * side;
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if idx >= max_idx {
            return Err(format!(
                "decoded index {} is outside bitmap bounds [0, {})",
                idx, max_idx
            ));
        }
        let x = (idx % side) as i64;
        let y = (idx / side) as i64;
        out.push((x, y));
    }
    Ok(out)
}

pub fn bitmap_bytes_for_exp(exp: u8) -> Result<usize, String> {
    if !(2..=15).contains(&exp) {
        return Err("bitmap exponent out of supported range [2, 15]".to_string());
    }
    Ok(1usize << (2 * exp as usize - 3))
}

pub(crate) fn set_lsb_bit(bytes: &mut [u8], idx: usize, value: bool) {
    let byte = idx / 8;
    let bit = idx % 8;
    let mask = 1u8 << bit;
    if value {
        bytes[byte] |= mask;
    } else {
        bytes[byte] &= !mask;
    }
}

pub(crate) fn test_lsb_bit(bytes: &[u8], idx: usize) -> bool {
    let byte = idx / 8;
    let bit = idx % 8;
    (bytes[byte] & (1u8 << bit)) != 0
}
