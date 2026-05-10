use super::{push_mercator_pixel, PixelType};
use crate::tile::GenericTile;
use crate::tile_archive::decompress_tile_block;
use crate::tile_range::{
    decompress_tile_range_response as core_decompress_tile_range_response, parse_tile_range_header,
    parse_tiles_from_body,
};
use crate::utils::{normalize_mercator_bounds, set_panic_hook};
use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// TODO: in later implementation, consider make cache in js side for better webgl integration.
const PIXEL_CACHE_MAX_ENTRIES: usize = 512;

#[wasm_bindgen]
/// Decoded tile container built from TileRangeResponse wire-format bytes.
///
/// The wire format itself is defined in `crate::tile_range`.
pub struct TileBuffer {
    pub(crate) tiles: Vec<(u16, u16, GenericTile)>,
    pub(crate) _level0_exp: u8,
    pub(crate) tile_grid_exp: u8,
    pub(crate) tile_bitmap_exp: u8,
    pub(crate) render_exp: u8,
    /// Cache of mercator pixel output keyed by (tile_x, tile_y, tile_z, render_exp, pixel_type).
    /// Uses RefCell for interior mutability since wasm_bindgen query methods take &self.
    mercator_cache: RefCell<HashMap<(u32, u32, u8, u8, PixelType), Vec<f32>>>,
}

#[wasm_bindgen]
impl TileBuffer {
    fn find_tile(&self, grid_x: u16, grid_y: u16) -> Option<&GenericTile> {
        self.tiles
            .iter()
            .find(|(x, y, _)| *x == grid_x && *y == grid_y)
            .map(|(_, _, tile)| tile)
    }

    fn clamped_query_render_exp(&self, tile_z: u8, requested_render_exp: u8) -> u8 {
        let world_detail_exp = self.tile_grid_exp as i16 + self.tile_bitmap_exp as i16;
        let max_render_exp = (world_detail_exp - tile_z as i16).max(0) as u8;
        requested_render_exp.min(max_render_exp)
    }

    /// Query tile buffer for pixels within a single tile(subtile or tile).
    fn query_tile_pixels_internal(
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

        // Coarse query: requested resolution is below the subtile grid resolution.
        // Reduce each internal tile to occupancy and OR into coarse output pixels.
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
                if tile.iter_pixels(0, 0, 0, 0, 0, 0).next().is_none() {
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

    /// Convert tile pixels to mercator coordinates.
    fn query_tile_mercator_pixels_internal(
        &self,
        tile_x: u32,
        tile_y: u32,
        tile_z: u8,
        render_exp: u8,
        pixel_type: PixelType,
    ) -> Vec<f32> {
        let render_exp = self.clamped_query_render_exp(tile_z, render_exp);

        let cache_key = (tile_x, tile_y, tile_z, render_exp, pixel_type);
        if let Some(cached) = self.mercator_cache.borrow().get(&cache_key) {
            return cached.clone();
        }

        log::info!(
            "cache missed: tile_x={}, tile_y={}, tile_z={}, render_exp={}, pixel_type={:?}",
            tile_x,
            tile_y,
            tile_z,
            render_exp,
            pixel_type
        );

        let packed_pixels = self.query_tile_pixels_internal(tile_x, tile_y, tile_z, render_exp);
        if packed_pixels.is_empty() {
            return Vec::new();
        }

        let Some(tiles_per_axis) = 1u32.checked_shl(tile_z as u32) else {
            return Vec::new();
        };
        let tile_world_size = 1.0 / tiles_per_axis as f64;
        let tile_merc_x0 = tile_x as f64 * tile_world_size;
        let tile_merc_y0 = tile_y as f64 * tile_world_size;
        let pixel_world_size = tile_world_size / (1u32 << render_exp) as f64;

        let mut mercator_pixels = Vec::with_capacity(packed_pixels.len() * 2);
        let mut idx = 0usize;
        while idx + 1 < packed_pixels.len() {
            let px = packed_pixels[idx] as f64;
            let py = packed_pixels[idx + 1] as f64;
            let merc_x = tile_merc_x0 + px * pixel_world_size;
            let merc_y = tile_merc_y0 + py * pixel_world_size;
            push_mercator_pixel(
                &mut mercator_pixels,
                pixel_type,
                merc_x,
                merc_y,
                pixel_world_size,
            );
            idx += 2;
        }

        let mut cache = self.mercator_cache.borrow_mut();
        if cache.len() >= PIXEL_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        cache.insert(cache_key, mercator_pixels.clone());

        mercator_pixels
    }

    /// Split range query into tile queries and merge the results.
    fn query_range_mercator_pixels_internal(
        &self,
        x: u32,
        y: u32,
        z: u8,
        w: u32,
        h: u32,
        render_exp: u8,
        pixel_type: PixelType,
    ) -> Vec<f32> {
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
                    .query_tile_mercator_pixels_internal(tile_x, tile_y, z, render_exp, pixel_type);
                out.extend_from_slice(&tile_pixels);
            }
        }
        out
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
        let tiles = parse_tiles_from_body(
            header.tile_bitmap_exp,
            header.x0,
            header.y0,
            header.range_w as usize,
            header.tile_count as usize,
            header.present_count as usize,
            body,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to parse TileRangeResponse: {}", e)))?;
        Ok(TileBuffer {
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
        self.tiles.len() as u32
    }

    #[wasm_bindgen]
    pub fn add_tile(
        &mut self,
        x: u16,
        y: u16,
        tile_bitmap_exp: u8,
        decompressed_block: &[u8],
    ) -> Result<(), JsValue> {
        if self.tile_bitmap_exp != tile_bitmap_exp {
            return Err(JsValue::from_str(
                "tile_bitmap_exp mismatch when adding tile to TileBuffer",
            ));
        }
        let tile = GenericTile::from_bytes(decompressed_block)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse decompressed tile: {e}")))?;
        self.tiles.push((x, y, tile));
        self.mercator_cache.borrow_mut().clear();
        Ok(())
    }

    #[wasm_bindgen]
    pub fn clear_cache(&self) {
        self.mercator_cache.borrow_mut().clear();
    }

    #[wasm_bindgen]
    pub fn total_pixel_count(&self) -> u32 {
        let mut count = 0u32;
        for (_, _, tile) in &self.tiles {
            count += tile
                .iter_pixels(0, 0, 0, 0, 0, self.tile_bitmap_exp as i16)
                .count() as u32;
        }
        count
    }

    fn get_bounding_mercator_pixels_internal(
        &self,
        sw_x: f64,
        sw_y: f64,
        ne_x: f64,
        ne_y: f64,
        pixel_type: PixelType,
    ) -> Vec<f32> {
        let (merc_x_min, merc_y_min, merc_x_max, merc_y_max) =
            normalize_mercator_bounds(sw_x, sw_y, ne_x, ne_y);

        let render_exp = self.render_exp.min(self.tile_bitmap_exp);
        let tile_grid_size = (1u32 << self.tile_grid_exp) as f64;
        let tile_world_size = 1.0 / tile_grid_size;
        let tile_pixel_size = tile_world_size / (1u32 << render_exp) as f64;

        let mut pixels = Vec::new();
        for (tx, ty, tile) in &self.tiles {
            let merc_x0 = *tx as f64 * tile_world_size;
            let merc_y0 = *ty as f64 * tile_world_size;
            let merc_x1 = merc_x0 + tile_world_size;
            let merc_y1 = merc_y0 + tile_world_size;

            if merc_x1 < merc_x_min
                || merc_x0 > merc_x_max
                || merc_y1 < merc_y_min
                || merc_y0 > merc_y_max
            {
                continue;
            }

            for (px, py) in tile.iter_pixels(0, 0, 0, 0, 0, render_exp as i16) {
                let merc_x = merc_x0 + px as f64 * tile_pixel_size;
                let merc_y = merc_y0 + py as f64 * tile_pixel_size;
                push_mercator_pixel(&mut pixels, pixel_type, merc_x, merc_y, tile_pixel_size);
            }
        }
        pixels
    }

    #[wasm_bindgen]
    pub fn get_bounding_mercator_pixels(
        &self,
        sw_x: f64,
        sw_y: f64,
        ne_x: f64,
        ne_y: f64,
    ) -> Vec<f32> {
        self.get_bounding_mercator_pixels_internal(sw_x, sw_y, ne_x, ne_y, PixelType::Pixel32)
    }

    #[wasm_bindgen]
    pub fn get_bounding_mercator_pixels64(
        &self,
        sw_x: f64,
        sw_y: f64,
        ne_x: f64,
        ne_y: f64,
    ) -> Vec<f32> {
        self.get_bounding_mercator_pixels_internal(sw_x, sw_y, ne_x, ne_y, PixelType::Pixel64)
    }

    #[wasm_bindgen]
    pub fn get_bounding_mercator_triangles(
        &self,
        sw_x: f64,
        sw_y: f64,
        ne_x: f64,
        ne_y: f64,
    ) -> Vec<f32> {
        self.get_bounding_mercator_pixels_internal(sw_x, sw_y, ne_x, ne_y, PixelType::Triangle64)
    }

    #[wasm_bindgen]
    pub fn get_tile_pixels(
        &self,
        tile_x: i64,
        tile_y: i64,
        tile_z: i16,
        buffer_size_power: i16,
    ) -> Vec<u16> {
        let _ = tile_z;
        if !(0..=u16::MAX as i64).contains(&tile_x) || !(0..=u16::MAX as i64).contains(&tile_y) {
            return Vec::new();
        }
        let tx = tile_x as u16;
        let ty = tile_y as u16;
        let Some(tile) = self.find_tile(tx, ty) else {
            return Vec::new();
        };

        let mut packed = Vec::new();
        for (px, py) in tile.iter_pixels(0, 0, 0, 0, 0, buffer_size_power) {
            if (0..=u16::MAX as i64).contains(&px) && (0..=u16::MAX as i64).contains(&py) {
                packed.push(px as u16);
                packed.push(py as u16);
            }
        }
        packed
    }

    #[wasm_bindgen]
    pub fn query_tile_pixels(
        &self,
        tile_x: u32,
        tile_y: u32,
        tile_z: u8,
        render_exp: u8,
    ) -> Vec<u16> {
        self.query_tile_pixels_internal(tile_x, tile_y, tile_z, render_exp)
    }

    #[wasm_bindgen]
    pub fn query_tile_mercator_pixels(
        &self,
        tile_x: u32,
        tile_y: u32,
        tile_z: u8,
        render_exp: u8,
    ) -> Vec<f32> {
        self.query_tile_mercator_pixels_internal(
            tile_x,
            tile_y,
            tile_z,
            render_exp,
            PixelType::Pixel32,
        )
    }

    #[wasm_bindgen]
    pub fn query_tile_mercator_pixels64(
        &self,
        tile_x: u32,
        tile_y: u32,
        tile_z: u8,
        render_exp: u8,
    ) -> Vec<f32> {
        self.query_tile_mercator_pixels_internal(
            tile_x,
            tile_y,
            tile_z,
            render_exp,
            PixelType::Pixel64,
        )
    }

    #[wasm_bindgen]
    pub fn query_tile_mercator_triangles(
        &self,
        tile_x: u32,
        tile_y: u32,
        tile_z: u8,
        render_exp: u8,
    ) -> Vec<f32> {
        self.query_tile_mercator_pixels_internal(
            tile_x,
            tile_y,
            tile_z,
            render_exp,
            PixelType::Triangle64,
        )
    }

    #[wasm_bindgen]
    pub fn query_range_mercator_pixels(
        &self,
        x: u32,
        y: u32,
        z: u8,
        w: u32,
        h: u32,
        render_exp: u8,
    ) -> Vec<f32> {
        self.query_range_mercator_pixels_internal(x, y, z, w, h, render_exp, PixelType::Pixel32)
    }

    #[wasm_bindgen]
    pub fn query_range_mercator_pixels64(
        &self,
        x: u32,
        y: u32,
        z: u8,
        w: u32,
        h: u32,
        render_exp: u8,
    ) -> Vec<f32> {
        self.query_range_mercator_pixels_internal(x, y, z, w, h, render_exp, PixelType::Pixel64)
    }

    #[wasm_bindgen]
    pub fn query_range_mercator_triangles(
        &self,
        x: u32,
        y: u32,
        z: u8,
        w: u32,
        h: u32,
        render_exp: u8,
    ) -> Vec<f32> {
        self.query_range_mercator_pixels_internal(x, y, z, w, h, render_exp, PixelType::Triangle64)
    }
}

#[wasm_bindgen]
pub fn decompress_tile_range_response(data: &[u8]) -> Result<Vec<u8>, JsValue> {
    core_decompress_tile_range_response(data)
        .map_err(|e| JsValue::from_str(&format!("Failed to decompress TileRangeResponse: {e}")))
}

#[wasm_bindgen]
pub fn decompress_fta_tile_block(compression: u8, data: &[u8]) -> Result<Vec<u8>, JsValue> {
    decompress_tile_block(compression, data)
        .map_err(|e| JsValue::from_str(&format!("Failed to decompress tile block: {e}")))
}
