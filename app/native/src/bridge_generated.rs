#![allow(
    non_camel_case_types,
    unused,
    clippy::redundant_closure,
    clippy::useless_conversion,
    clippy::unit_arg,
    clippy::double_parens,
    non_snake_case,
    clippy::too_many_arguments
)]
// AUTO GENERATED FILE, DO NOT EDIT.
// Generated by `flutter_rust_bridge`@ 1.82.3.

use crate::api::*;
use core::panic::UnwindSafe;
use flutter_rust_bridge::rust2dart::IntoIntoDart;
use flutter_rust_bridge::*;
use std::ffi::c_void;
use std::sync::Arc;

// Section: imports

use crate::map_renderer::RenderResult;
use crate::storage::RawDataFile;

// Section: wire functions

fn wire_init_impl(
    port_: MessagePort,
    temp_dir: impl Wire2Api<String> + UnwindSafe,
    doc_dir: impl Wire2Api<String> + UnwindSafe,
    support_dir: impl Wire2Api<String> + UnwindSafe,
    cache_dir: impl Wire2Api<String> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, (), _>(
        WrapInfo {
            debug_name: "init",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_temp_dir = temp_dir.wire2api();
            let api_doc_dir = doc_dir.wire2api();
            let api_support_dir = support_dir.wire2api();
            let api_cache_dir = cache_dir.wire2api();
            move |task_callback| {
                Result::<_, ()>::Ok(init(
                    api_temp_dir,
                    api_doc_dir,
                    api_support_dir,
                    api_cache_dir,
                ))
            }
        },
    )
}
fn wire_render_map_overlay_impl(
    port_: MessagePort,
    zoom: impl Wire2Api<f32> + UnwindSafe,
    left: impl Wire2Api<f64> + UnwindSafe,
    top: impl Wire2Api<f64> + UnwindSafe,
    right: impl Wire2Api<f64> + UnwindSafe,
    bottom: impl Wire2Api<f64> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, Option<RenderResult>, _>(
        WrapInfo {
            debug_name: "render_map_overlay",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_zoom = zoom.wire2api();
            let api_left = left.wire2api();
            let api_top = top.wire2api();
            let api_right = right.wire2api();
            let api_bottom = bottom.wire2api();
            move |task_callback| {
                Result::<_, ()>::Ok(render_map_overlay(
                    api_zoom, api_left, api_top, api_right, api_bottom,
                ))
            }
        },
    )
}
fn wire_on_location_update_impl(
    port_: MessagePort,
    latitude: impl Wire2Api<f64> + UnwindSafe,
    longitude: impl Wire2Api<f64> + UnwindSafe,
    timestamp_ms: impl Wire2Api<i64> + UnwindSafe,
    accuracy: impl Wire2Api<f32> + UnwindSafe,
    altitude: impl Wire2Api<Option<f32>> + UnwindSafe,
    speed: impl Wire2Api<Option<f32>> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, (), _>(
        WrapInfo {
            debug_name: "on_location_update",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_latitude = latitude.wire2api();
            let api_longitude = longitude.wire2api();
            let api_timestamp_ms = timestamp_ms.wire2api();
            let api_accuracy = accuracy.wire2api();
            let api_altitude = altitude.wire2api();
            let api_speed = speed.wire2api();
            move |task_callback| {
                Result::<_, ()>::Ok(on_location_update(
                    api_latitude,
                    api_longitude,
                    api_timestamp_ms,
                    api_accuracy,
                    api_altitude,
                    api_speed,
                ))
            }
        },
    )
}
fn wire_list_all_raw_data_impl(port_: MessagePort) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, Vec<RawDataFile>, _>(
        WrapInfo {
            debug_name: "list_all_raw_data",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || move |task_callback| Result::<_, ()>::Ok(list_all_raw_data()),
    )
}
fn wire_get_raw_data_mode_impl(port_: MessagePort) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, bool, _>(
        WrapInfo {
            debug_name: "get_raw_data_mode",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || move |task_callback| Result::<_, ()>::Ok(get_raw_data_mode()),
    )
}
fn wire_toggle_raw_data_mode_impl(port_: MessagePort, enable: impl Wire2Api<bool> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, (), _>(
        WrapInfo {
            debug_name: "toggle_raw_data_mode",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_enable = enable.wire2api();
            move |task_callback| Result::<_, ()>::Ok(toggle_raw_data_mode(api_enable))
        },
    )
}
fn wire_finalize_ongoing_journey_impl(port_: MessagePort) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap::<_, _, _, (), _>(
        WrapInfo {
            debug_name: "finalize_ongoing_journey",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || move |task_callback| Result::<_, ()>::Ok(finalize_ongoing_journey()),
    )
}
// Section: wrapper structs

// Section: static checks

// Section: allocate functions

// Section: related functions

// Section: impl Wire2Api

pub trait Wire2Api<T> {
    fn wire2api(self) -> T;
}

impl<T, S> Wire2Api<Option<T>> for *mut S
where
    *mut S: Wire2Api<T>,
{
    fn wire2api(self) -> Option<T> {
        (!self.is_null()).then(|| self.wire2api())
    }
}

impl Wire2Api<bool> for bool {
    fn wire2api(self) -> bool {
        self
    }
}

impl Wire2Api<f32> for f32 {
    fn wire2api(self) -> f32 {
        self
    }
}
impl Wire2Api<f64> for f64 {
    fn wire2api(self) -> f64 {
        self
    }
}
impl Wire2Api<i64> for i64 {
    fn wire2api(self) -> i64 {
        self
    }
}

impl Wire2Api<u8> for u8 {
    fn wire2api(self) -> u8 {
        self
    }
}

// Section: impl IntoDart

impl support::IntoDart for RawDataFile {
    fn into_dart(self) -> support::DartAbi {
        vec![
            self.name.into_into_dart().into_dart(),
            self.path.into_into_dart().into_dart(),
        ]
        .into_dart()
    }
}
impl support::IntoDartExceptPrimitive for RawDataFile {}
impl rust2dart::IntoIntoDart<RawDataFile> for RawDataFile {
    fn into_into_dart(self) -> Self {
        self
    }
}

impl support::IntoDart for RenderResult {
    fn into_dart(self) -> support::DartAbi {
        vec![
            self.left.into_into_dart().into_dart(),
            self.top.into_into_dart().into_dart(),
            self.right.into_into_dart().into_dart(),
            self.bottom.into_into_dart().into_dart(),
            self.data.into_into_dart().into_dart(),
        ]
        .into_dart()
    }
}
impl support::IntoDartExceptPrimitive for RenderResult {}
impl rust2dart::IntoIntoDart<RenderResult> for RenderResult {
    fn into_into_dart(self) -> Self {
        self
    }
}

// Section: executor

support::lazy_static! {
    pub static ref FLUTTER_RUST_BRIDGE_HANDLER: support::DefaultHandler = Default::default();
}

/// cbindgen:ignore
#[cfg(target_family = "wasm")]
#[path = "bridge_generated.web.rs"]
mod web;
#[cfg(target_family = "wasm")]
pub use web::*;

#[cfg(not(target_family = "wasm"))]
#[path = "bridge_generated.io.rs"]
mod io;
#[cfg(not(target_family = "wasm"))]
pub use io::*;