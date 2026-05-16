use super::PixelType;
use crate::bitmap2d::BitMap2D;
use crate::tile_range::{
    decompress_tile_range_response as core_decompress_tile_range_response, parse_tile_range_header,
    parse_tiles_from_body,
};
use crate::utils::set_panic_hook;
use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
/// Decoded tile container built from TileRangeResponse wire-format bytes.
/// TileBuffer stores a set of tiles, and proxy the queries the requests to the tiles.
///   TileBuffer allows two groups of queries: 
///   - get_tile_pixels: get pixel coordinates within a single tile(subtile or tile).
///   - query_range_mercator_pixels: query pixels within a range of tiles.
///
/// The wire format itself is defined in `crate::tile_range`.
pub struct TileBuffer {
    pub(crate) grid_origin_x: u16,
    pub(crate) grid_origin_y: u16,
    pub(crate) grid_w: u16,
    pub(crate) grid_h: u16,
    /// Row-major grid: index = (y - grid_origin_y) * grid_w + (x - grid_origin_x).
    /// Absent tiles are `None`.
    pub(crate) tiles: Vec<Option<BitMap2D>>,
    pub(crate) _level0_exp: u8,
    pub(crate) tile_grid_exp: u8,
    pub(crate) tile_bitmap_exp: u8,
    pub(crate) render_exp: u8,
    /// Cache of mercator pixel output keyed by (tile_x, tile_y, tile_z, render_exp, pixel_type).
    /// Uses RefCell for interior mutability since wasm_bindgen query methods take &self.
    pub(crate) mercator_cache: RefCell<HashMap<(u32, u32, u8, u8, PixelType), Vec<f32>>>,
}

#[wasm_bindgen]
impl TileBuffer {
    pub(crate) fn find_tile(&self, grid_x: u16, grid_y: u16) -> Option<&BitMap2D> {
        let dx = grid_x.checked_sub(self.grid_origin_x)? as usize;
        let dy = grid_y.checked_sub(self.grid_origin_y)? as usize;
        if dx >= self.grid_w as usize || dy >= self.grid_h as usize {
            return None;
        }
        self.tiles[dy * self.grid_w as usize + dx].as_ref()
    }

    pub(crate) fn clamped_query_render_exp(&self, tile_z: u8, requested_render_exp: u8) -> u8 {
        let world_detail_exp = self.tile_grid_exp as i16 + self.tile_bitmap_exp as i16;
        let max_render_exp = (world_detail_exp - tile_z as i16).max(0) as u8;
        requested_render_exp.min(max_render_exp)
    }

    #[wasm_bindgen]
    /// Query tile buffer for pixels within a single tile(subtile or tile).
    pub fn get_tile_pixels(
        &self,
        tile_x: u32,
        tile_y: u32,
        tile_z: u8,
        render_exp: u8,
    ) -> Vec<u16> {
        let Some(tiles_per_axis) = 1u32.checked_shl(tile_z as u32) else {
            return Vec::new();
        };
        if tile_x >= tiles_per_axis || tile_y >= tiles_per_axis {
            return Vec::new();
        }

        let render_exp = self.clamped_query_render_exp(tile_z, render_exp);
        let mut packed = Vec::new();

        if tile_z >= self.tile_grid_exp {
            // Case 1: The queried tiles are smaller than the TileBuffer's internal tile grid.
            let dz = tile_z - self.tile_grid_exp;
            let parent_x = tile_x >> dz;
            let parent_y = tile_y >> dz;
            if parent_x > u16::MAX as u32 || parent_y > u16::MAX as u32 {
                return Vec::new();
            }

            let Some(tile) = self.find_tile(parent_x as u16, parent_y as u16) else {
                return Vec::new();
            };

            let child_mask = if dz == 0 { 0 } else { (1u32 << dz) - 1 };
            let child_x = (tile_x & child_mask) as i64;
            let child_y = (tile_y & child_mask) as i64;
            let child_z = dz as i16;
            for (px, py) in tile.iter_pixels(0, 0, child_x, child_y, child_z, render_exp as i16) {
                if (0..=u16::MAX as i64).contains(&px) && (0..=u16::MAX as i64).contains(&py) {
                    packed.push(px as u16);
                    packed.push(py as u16);
                }
            }
            return packed;
        }

        let span = self.tile_grid_exp - tile_z;
        let subtiles_per_axis = 1u32 << span;
        let base_x = tile_x << span;
        let base_y = tile_y << span;

        if render_exp >= span {
            // Case 2: The queried tiles are larger than the TileBuffer's internal tile grid.
            // TODO(opt): with the grid-indexed layout we could iterate the tiles slice
            // directly instead of calling find_tile per (dx, dy).
            let sub_render_exp = render_exp - span;
            for dy in 0..subtiles_per_axis {
                for dx in 0..subtiles_per_axis {
                    let gx = base_x + dx;
                    let gy = base_y + dy;
                    if gx > u16::MAX as u32 || gy > u16::MAX as u32 {
                        continue;
                    }
                    let Some(tile) = self.find_tile(gx as u16, gy as u16) else {
                        continue;
                    };
                    for (px, py) in tile.iter_pixels(0, 0, 0, 0, 0, sub_render_exp as i16) {
                        let out_x = (dx << sub_render_exp) + px as u32;
                        let out_y = (dy << sub_render_exp) + py as u32;
                        if out_x <= u16::MAX as u32 && out_y <= u16::MAX as u32 {
                            packed.push(out_x as u16);
                            packed.push(out_y as u16);
                        }
                    }
                }
            }
            return packed;
        }

        // Case 3: The requested resolution is below the subtile grid resolution.
        // Reduce each internal tile to occupancy and OR into coarse output pixels.
        // TODO(opt): same as Case 2 — direct grid slice iteration would avoid per-cell find_tile.
        let coarse_shift = span - render_exp;
        for dy in 0..subtiles_per_axis {
            for dx in 0..subtiles_per_axis {
                let gx = base_x + dx;
                let gy = base_y + dy;
                if gx > u16::MAX as u32 || gy > u16::MAX as u32 {
                    continue;
                }
                let Some(tile) = self.find_tile(gx as u16, gy as u16) else {
                    continue;
                };
                if tile.is_empty() {
                    continue;
                }
                let out_x = dx >> coarse_shift;
                let out_y = dy >> coarse_shift;
                if out_x <= u16::MAX as u32 && out_y <= u16::MAX as u32 {
                    packed.push(out_x as u16);
                    packed.push(out_y as u16);
                }
            }
        }

        packed
    }

    #[wasm_bindgen]
    /// Parses raw TileRangeResponse bytes returned by the `/tile-range` endpoint.
    ///
    /// `data` must match the binary format documented in `crate::tile_range`.
    pub fn new_from_tile_range_response(
        level0_exp: u8,
        data: &[u8],
    ) -> Result<TileBuffer, JsValue> {
        set_panic_hook();
        let decompressed = core_decompress_tile_range_response(data).map_err(|e| {
            JsValue::from_str(&format!("Failed to decompress TileRangeResponse: {}", e))
        })?;
        let header = parse_tile_range_header(&decompressed)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse TileRange header: {}", e)))?;
        let body = &decompressed[16..];
        let parsed = parse_tiles_from_body(
            header.tile_bitmap_exp,
            header.x0,
            header.y0,
            header.range_w as usize,
            header.tile_count as usize,
            header.present_count as usize,
            body,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to parse TileRangeResponse: {}", e)))?;
        let grid_w = header.range_w;
        let grid_h = header.range_h;
        let mut tiles = vec![None; grid_w as usize * grid_h as usize];
        for (x, y, bm) in parsed {
            let idx = (y - header.y0) as usize * grid_w as usize + (x - header.x0) as usize;
            tiles[idx] = Some(bm);
        }
        Ok(TileBuffer {
            grid_origin_x: header.x0,
            grid_origin_y: header.y0,
            grid_w,
            grid_h,
            tiles,
            _level0_exp: level0_exp,
            tile_grid_exp: header.z,
            tile_bitmap_exp: header.tile_bitmap_exp,
            render_exp: header.tile_bitmap_exp,
            mercator_cache: RefCell::new(HashMap::new()),
        })
    }

    #[wasm_bindgen]
    pub fn set_render_exp(&mut self, exp: u8) {
        self.render_exp = exp;
    }

    #[wasm_bindgen]
    pub fn tile_count(&self) -> u32 {
        self.tiles.iter().filter(|t| t.is_some()).count() as u32
    }

    #[wasm_bindgen]
    pub fn total_pixel_count(&self) -> u32 {
        let mut count = 0u32;
        for bm in self.tiles.iter().filter_map(|t| t.as_ref()) {
            count += bm
                .iter_pixels(0, 0, 0, 0, 0, self.tile_bitmap_exp as i16)
                .count() as u32;
        }
        count
    }

    /// Split range query into tile queries and merge the results.
    #[wasm_bindgen]
    pub fn query_range_pixels(
        &self,
        x: u32,
        y: u32,
        z: u8,
        w: u32,
        h: u32,
        render_exp: u8
    ) -> Vec<u16> {
        if w == 0 || h == 0 {
            return Vec::new();
        }

        let mut out = Vec::new();
        for dy in 0..h {
            for dx in 0..w {
                let Some(tile_x) = x.checked_add(dx) else {
                    continue;
                };
                let Some(tile_y) = y.checked_add(dy) else {
                    continue;
                };
                let tile_pixels = self
                    .get_tile_pixels(tile_x.into(), tile_y.into(), z.into(), render_exp.into());
                out.extend_from_slice(&tile_pixels);
            }
        }
        out
    }

}

#[wasm_bindgen]
pub fn decompress_tile_range_response(data: &[u8]) -> Result<Vec<u8>, JsValue> {
    core_decompress_tile_range_response(data)
        .map_err(|e| JsValue::from_str(&format!("Failed to decompress TileRangeResponse: {e}")))
}
