use crate::tile::{xy_to_index, GenericTile};
use bitvec::prelude::*;

/// There are two common use case for accessing the tiles:
/// 1. Iterate over the pixels of the tile(subtile) at a given resolution.
/// 2. Iterate over the pixels as mercator coordinates.

/// IndexIter helps index pixels within a tile with a specific width_exp.
pub struct IndexIter {
    x_min: i64,
    x_max: i64,
    y_min: i64,
    y_max: i64,
    current_x: i64,
    current_y: i64,
}

impl Iterator for IndexIter {
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_y >= self.y_max {
            return None;
        }

        let index = (self.current_x, self.current_y);

        self.current_x += 1;
        if self.current_x >= self.x_max {
            self.current_x = self.x_min;
            self.current_y += 1;
        }

        Some(index)
    }
}

impl IndexIter {
    /// the interest region of a tile
    fn new(x: i64, y: i64, resolution_exp: i16) -> Self {
        let x_min = x << resolution_exp;
        let x_max = (x + 1) << resolution_exp;
        let y_min = y << resolution_exp;
        let y_max = (y + 1) << resolution_exp;

        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            current_x: x_min,
            current_y: y << resolution_exp,
        }
    }

    fn get_min_xy(&self) -> (i64, i64) {
        (self.x_min, self.y_min)
    }
}

pub struct MipmapIter<'a> {
    bitmap: &'a BitVec,
    index_iter: IndexIter,
    width_exp: i16,
    start_x: i64,
    start_y: i64,
    x_offset: i64,
    y_offset: i64,
}

impl<'a> Iterator for MipmapIter<'a> {
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((x, y)) = self.index_iter.next() {
            if self.bitmap[xy_to_index(x, y, self.width_exp)] {
                return Some((
                    self.start_x + x - self.x_offset,
                    self.start_y + y - self.y_offset,
                ));
            }
        }
        None
    }
}

impl<'a> MipmapIter<'a> {
    pub fn new(
        bitmap: &'a BitVec,
        start_x: i64,
        start_y: i64,
        x: i64,
        y: i64,
        z: i16,
        width_exp: i16,
    ) -> Self {
        let index_iter = IndexIter::new(x, y, width_exp - z);
        let (x_offset, y_offset) = index_iter.get_min_xy();
        Self {
            bitmap,
            index_iter,
            width_exp,
            start_x,
            start_y,
            x_offset,
            y_offset,
        }
    }
}

pub struct OverscanIter<'a> {
    bitmap: &'a BitVec,
    index_iter: IndexIter,
    sub_tile_index_iter: Option<IndexIter>,
    width_exp: i16,
    start_x: i64,
    start_y: i64,
    subtile_resolution_exp: i16,
}

impl<'a> Iterator for OverscanIter<'a> {
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.sub_tile_index_iter.is_none() {
                while let Some((x, y)) = self.index_iter.next() {
                    if self.bitmap[xy_to_index(x, y, self.width_exp)] {
                        let (x_min, y_min) = self.index_iter.get_min_xy();
                        self.sub_tile_index_iter = Some(IndexIter::new(
                            x - x_min,
                            y - y_min,
                            self.subtile_resolution_exp,
                        ));
                        break;
                    }
                }

                if self.sub_tile_index_iter.is_none() {
                    return None;
                }
            }

            if let Some((x, y)) = self.sub_tile_index_iter.as_mut().unwrap().next() {
                return Some((self.start_x + x, self.start_y + y));
            } else {
                self.sub_tile_index_iter = None;
            }
        }
    }
}

impl<'a> OverscanIter<'a> {
    pub fn new(
        bitmap: &'a BitVec,
        start_x: i64,
        start_y: i64,
        x: i64,
        y: i64,
        z: i16,
        width_exp: i16,
        subtile_resolution_exp: i16,
    ) -> Self {
        Self {
            bitmap,
            // TODO: check this line
            index_iter: IndexIter::new(x, y, width_exp - z),
            sub_tile_index_iter: None,
            width_exp,
            start_x,
            start_y,
            subtile_resolution_exp,
        }
    }
}

/// SubtileIter iterates over the pixels of the subtile at a given resolution.
/// width_exp is the width of the large tile
/// start_x, start_y is the coordinate of the top left corner of the target image
/// subtile_resolution_exp is the resolution for each subtile
/// index_iter will iterate over the subtile indices of the region of interest
/// Slots that are `None` are skipped intentionally. In partial-loading mode,
/// callers should run `GenericTile::check_ready` first to detect missing
/// subtiles before relying on complete results.
pub struct SubtileIter<'a> {
    sub_tiles: &'a [Option<Box<GenericTile>>],
    current_sub_tile_iter: Option<Box<TilePixelIter<'a>>>,
    index_iter: IndexIter,
    width_exp: i16,
    start_x: i64,
    start_y: i64,
    subtile_resolution_exp: i16,
}

impl<'a> Iterator for SubtileIter<'a> {
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        // 2. if subtile iter available, use it until it's exhausted.
        // 2.1 if it's exhauseted, remove it

        loop {
            // if no subtile iter available, scan for the next non-empty sub tile
            if self.current_sub_tile_iter.is_none() {
                while let Some((x, y)) = self.index_iter.next() {
                    // println!("iter x, y: {}, {}, self.start_x: {}, self.start_y: {}, self.width_exp: {}， subtile_resolution_exp: {}", x, y, self.start_x, self.start_y, self.width_exp, self.subtile_resolution_exp);
                    if let Some(sub_tile) = &self.sub_tiles[xy_to_index(x, y, self.width_exp)] {
                        // let sub_tile_resolution_exp =

                        let (x_min, y_min) = self.index_iter.get_min_xy(); // println!("self.start_x: {}, x-x_min: {}, self.subtile_resolution_exp: {}", self.start_x, x-x_min, self.subtile_resolution_exp);
                        let start_x = self.start_x + ((x - x_min) << self.subtile_resolution_exp);
                        let start_y = self.start_y + ((y - y_min) << self.subtile_resolution_exp);
                        // println!("start_x, start_y: {}, {}, x, y: {}, {}, x_min, y_min: {}, {}", start_x, start_y, x, y, x_min, y_min);
                        let iter = sub_tile.iter_pixels(
                            start_x,
                            start_y,
                            0,
                            0,
                            0,
                            self.subtile_resolution_exp,
                        );
                        self.current_sub_tile_iter = Some(Box::new(iter));
                        break;
                    }
                }
                if self.current_sub_tile_iter.is_none() {
                    // TODO: or just return None;?
                    break;
                }
            }

            if let Some((x, y)) = self.current_sub_tile_iter.as_mut().unwrap().next() {
                // return Some((self.start_x + x, self.start_y + y));
                return Some((x, y));
            } else {
                self.current_sub_tile_iter = None;
            }
        }

        return None;
    }
}

impl<'a> SubtileIter<'a> {
    pub fn new(
        sub_tiles: &'a [Option<Box<GenericTile>>],
        start_x: i64,
        start_y: i64,
        x: i64,
        y: i64,
        z: i16,
        width_exp: i16,
        subtile_resolution_exp: i16,
    ) -> Self {
        // println!("create subtile iter, start_x, start_y: {}, {}, xyz: {}, {}, {}, width_exp: {}, subtile_resolution_exp: {}", start_x, start_y, x, y, z, width_exp, subtile_resolution_exp);
        // let start_x = start_x - x << subtile_resolution_exp;
        // let start_y = start_y - y << subtile_resolution_exp;
        // let start_x = start_x - x << (width_exp - z + subtile_resolution_exp);
        // let start_y = start_y - y << (width_exp - z + subtile_resolution_exp);
        // let start_x = start_x - x << (width_exp - z);
        // let start_y = start_y - y << (width_exp - z);
        Self {
            sub_tiles,
            current_sub_tile_iter: None,
            index_iter: IndexIter::new(x, y, width_exp - z),
            width_exp,
            start_x,
            start_y,
            subtile_resolution_exp,
        }
    }
}

pub enum TilePixelIter<'a> {
    MipmapIter(MipmapIter<'a>),
    OverscanIter(OverscanIter<'a>),
    SubTileIter(SubtileIter<'a>),
    Empty,
}

impl<'a> Iterator for TilePixelIter<'a> {
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TilePixelIter::MipmapIter(iter) => iter.next(),
            TilePixelIter::OverscanIter(iter) => iter.next(),
            TilePixelIter::SubTileIter(iter) => iter.next(),
            TilePixelIter::Empty => None,
        }
    }
}
