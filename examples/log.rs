#[macro_use]
extern crate slog;
extern crate slog_term;

use slog::DrainExt;

fn main() {
    let drain = slog_term::streamer().build().fuse();
    let root_logger = slog::Logger::root(drain, o!("version" => "0.5"));
    info!(root_logger, "Evolution(s) started";
        "started_at" => format!("{}", time::now().rfc3339()));
}
