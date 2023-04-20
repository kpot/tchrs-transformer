use md5::{Digest, Md5};
use std::fmt::Write as _;
use std::{
    fs,
    io::{self, Write as _},
    path,
};

type FileHash = [u8; 16];

#[cfg(feature = "download")]
fn display_copy_progress(bytes_read: usize, bytes_total: usize, terminal_width: u16) {
    const BLANK_SPACES: u16 = 20;
    const TICK: &[u8; 1] = b"#";
    const SPACE: &[u8; 1] = b" ";
    let ticks_total = if BLANK_SPACES < terminal_width {
        terminal_width - BLANK_SPACES
    } else {
        1
    } as usize;
    let display_ticks = ticks_total as usize * bytes_read / bytes_total;
    {
        let mut stdout = io::stdout().lock();
        stdout.write_all(b"\r[").ok();
        for _ in 0..display_ticks {
            stdout.write_all(TICK).ok();
        }
        for _ in display_ticks..ticks_total {
            stdout.write_all(SPACE).ok();
        }
        stdout.write_all(b"]: ").ok();
        stdout.flush().ok();
    }
    print!("{:>15} B", bytes_read);
}

#[cfg(feature = "download")]
fn hash_file(path: impl AsRef<path::Path>) -> io::Result<String> {
    let mut file = fs::File::open(path.as_ref())?;
    let mut hasher = Md5::new();
    io::copy(&mut file, &mut hasher)?;
    let bin_hash: FileHash = hasher.finalize().into();
    let mut str_hash = String::with_capacity(2 * bin_hash.len());
    for byte in bin_hash.iter() {
        write!(&mut str_hash, "{:02x}", byte).ok();
    }
    Ok(str_hash)
}

/// An analog for std::io::copy, but with a nice progress bar.
#[cfg(feature = "download")]
fn copy_with_progress<W>(
    mut from: Box<dyn io::Read>,
    to: &mut W,
    total_size: Option<usize>,
) -> io::Result<usize>
where
    W: io::Write,
{
    let mut buffer = [0u8; 256];
    const TERMINAL_WIDTH: u16 = 70;
    let mut bytes_read_total = 0;
    loop {
        let bytes_read = from.read(&mut buffer[..])?;
        if bytes_read > 0 {
            to.write_all(&buffer[..bytes_read])?;
            bytes_read_total += bytes_read;
            if let Some(bytes_total) = total_size {
                display_copy_progress(bytes_read_total, bytes_total, TERMINAL_WIDTH);
            }
        } else {
            break;
        }
    }
    std::io::stdout().lock().write_all(b"\n").ok();
    Ok(bytes_read_total)
}

/// Dowloads a file from a URL if it's not already present in its target location
/// or its MD5 hash is different there.
/// Useful for fetching various datasets.
#[cfg(feature = "download")]
pub fn download(
    origin_url: impl AsRef<str>,
    hash: impl AsRef<str>,
    target: impl AsRef<path::Path>,
) -> io::Result<()> {
    let url = origin_url.as_ref();
    let path = target.as_ref();
    if let Some(parent_dir) = path.parent() {
        fs::create_dir_all(parent_dir)?;
    }
    let skip_download = path.exists() && hash.as_ref().to_lowercase() == hash_file(path)?;
    if skip_download {
        Ok(())
    } else {
        let response = ureq::get(url)
            .call()
            .map_err(|response_err| io::Error::new(io::ErrorKind::Other, response_err))?;
        if response.status() == 200 {
            let content_length = response
                .header("Content-Length")
                .and_then(|header_str| header_str.parse::<usize>().ok());

            let response_reader = response.into_reader();
            let mut target_file = fs::File::create(&path)?;
            println!("Downloading: {} as {}", url, path.to_string_lossy());
            copy_with_progress(response_reader, &mut target_file, content_length)?;
            print!("\n");
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid response code: {}", response.status()),
            ))
        }
    }
}
