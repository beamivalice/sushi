//! MLX's page grid for a buffer written in place. MLX rounds an allocation past a page up to whole pages, its pool
//! hands an allocation a free buffer up to two pages larger (`reuse_from_cache`), and it donates a buffer to a write
//! only when the buffer is at most one page past the array's bytes (`is_donatable`).

pub const page: u64 = 16384;

/// PURE: the rows a buffer appended in place holds for `rows`, where `part_bytes` is a row's bytes in each buffer that
/// grows with it: 256-row steps on which every part past a page is a whole number of pages, so whatever buffer MLX's
/// pool hands a part is one MLX donates, and the append writes in place.
pub fn pagedRows(rows: u64, part_bytes: []const u64) u64 {
    var capacity = (rows +| 255) / 256 * 256;
    while (!onPages(capacity, part_bytes)) capacity += 256;
    return capacity;
}

fn onPages(rows: u64, part_bytes: []const u64) bool {
    for (part_bytes) |b| if (rows *| b > page and rows *| b % page != 0) return false;
    return true;
}
