use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const REQUEST_MAGIC: [u8; 4] = *b"PMCR";
const RESPONSE_MAGIC: [u8; 4] = *b"PMCS";
const PROTOCOL_VERSION: u32 = 1;
const OP_PROBE: u32 = 0;
const OP_CONVERT: u32 = 1;
const FLAG_OPTIMIZE: u32 = 1 << 0;
const FLAG_GLOBALS_PRESENT: u32 = 1 << 1;
const ALLOWED_FLAGS: u32 = FLAG_OPTIMIZE | FLAG_GLOBALS_PRESENT;
const REQUEST_HEADER_BYTES: usize = 24;
const RESPONSE_HEADER_BYTES: usize = 16;

/// Maximum combined UTF-8 bytes accepted from `file_globals` + shader body.
/// Real MilkDrop shader bodies are normally tens of KiB; 1 MiB leaves generous
/// compatibility headroom while bounding allocation and legacy parser exposure.
pub const MAX_CONVERTER_INPUT_BYTES: usize = 1024 * 1024;
/// Maximum success/error payload accepted back from the helper.
pub const MAX_CONVERTER_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
/// Wall-clock deadline for one native conversion helper.
pub const CONVERTER_TIMEOUT: Duration = Duration::from_secs(8);

const MAX_STDERR_BYTES: usize = 64 * 1024;
const HELPER_FILE_STEM: &str = "particle-milkdrop-converter-helper";
/// Hidden first argument used by executables that self-host helper mode.
pub const CURRENT_EXE_HELPER_ARG: &str = "--particle-milkdrop-converter-helper";
/// Exact-path override for the dedicated converter helper. When set, this is
/// authoritative: invalid or empty values fail closed without trying another
/// discovery mechanism.
pub const HELPER_PATH_ENV: &str = "PARTICLE_MILK_CONVERTER_HELPER";

#[derive(Clone, Debug)]
struct HelperSpec {
    path: PathBuf,
    args: Vec<OsString>,
}

impl HelperSpec {
    fn dedicated(path: PathBuf) -> Self {
        Self {
            path,
            args: Vec::new(),
        }
    }

    fn self_hosted(path: PathBuf) -> Self {
        Self {
            path,
            args: vec![OsString::from(CURRENT_EXE_HELPER_ARG)],
        }
    }
}

static SELF_HOSTED_HELPER: OnceLock<PathBuf> = OnceLock::new();

enum Request<'a> {
    Probe,
    Convert {
        file_globals: Option<&'a str>,
        body: &'a str,
        optimize: bool,
    },
}

struct DecodedConvertRequest {
    file_globals: Option<String>,
    body: String,
    optimize: bool,
}

enum DecodedRequest {
    Probe,
    Convert(DecodedConvertRequest),
}

/// Register the current executable as a helper host. The executable must call
/// [`current_executable_helper_mode`] before starting threads or parsing its
/// normal CLI. OjoDrop uses this contract so its bundled app stays a single
/// executable; other embedders may instead deploy the dedicated helper beside
/// their executable or set [`HELPER_PATH_ENV`] to its exact path.
pub fn register_current_executable_as_helper() -> Result<(), String> {
    if !crate::NATIVE_CONVERTER_AVAILABLE {
        return Err("this executable was built without the native converter payload".to_string());
    }
    let path = std::env::current_exe()
        .map_err(|e| format!("cannot resolve current executable for converter helper: {e}"))?;
    if let Some(existing) = SELF_HOSTED_HELPER.get() {
        return if existing == &path {
            Ok(())
        } else {
            Err(format!(
                "converter helper already registered as {}",
                existing.display()
            ))
        };
    }
    SELF_HOSTED_HELPER
        .set(path)
        .map_err(|_| "converter helper registration raced".to_string())
}

/// Return a helper exit status when this process was launched in the hidden
/// one-shot helper mode. Call this at the very start of an executable that used
/// [`register_current_executable_as_helper`] in its normal mode.
pub fn current_executable_helper_mode() -> Option<i32> {
    let requested = std::env::args_os()
        .nth(1)
        .is_some_and(|arg| arg == OsStr::new(CURRENT_EXE_HELPER_ARG));
    requested.then(run_converter_helper_stdio)
}

/// Probe the configured helper subprocess. This is deliberately runtime-aware:
/// compiling/linking the C++ payload is not enough if an embedding application
/// neither self-registers nor packages a discoverable helper executable.
pub fn helper_available() -> bool {
    let Ok(spec) = resolve_helper() else {
        return false;
    };
    matches!(
        invoke_helper(&spec, Request::Probe, Duration::from_secs(1)),
        Ok(response) if response == "available"
    )
}

pub(crate) fn convert(
    file_globals: Option<&str>,
    body: &str,
    optimize: bool,
) -> Result<String, String> {
    validate_input_size(file_globals, body)?;
    let spec = resolve_helper()?;
    invoke_helper(
        &spec,
        Request::Convert {
            file_globals,
            body,
            optimize,
        },
        CONVERTER_TIMEOUT,
    )
}

fn validate_input_size(file_globals: Option<&str>, body: &str) -> Result<(), String> {
    let globals_len = file_globals.map_or(0, str::len);
    let total = globals_len
        .checked_add(body.len())
        .ok_or_else(|| "native converter input length overflow".to_string())?;
    if total > MAX_CONVERTER_INPUT_BYTES {
        return Err(format!(
            "native converter input is {total} bytes, exceeding the {}-byte safety cap",
            MAX_CONVERTER_INPUT_BYTES
        ));
    }
    Ok(())
}

fn resolve_helper() -> Result<HelperSpec, String> {
    if let Some(configured) = std::env::var_os(HELPER_PATH_ENV) {
        if configured.is_empty() {
            return Err(format!("{HELPER_PATH_ENV} is set but empty"));
        }
        let path = PathBuf::from(configured);
        if !path.is_file() {
            return Err(format!(
                "{HELPER_PATH_ENV} points to missing helper {}",
                path.display()
            ));
        }
        return Ok(HelperSpec::dedicated(path));
    }

    if let Some(path) = SELF_HOSTED_HELPER.get() {
        return Ok(HelperSpec::self_hosted(path.clone()));
    }

    let current = std::env::current_exe()
        .map_err(|e| format!("cannot locate converter helper beside current executable: {e}"))?;
    let adjacent = current
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(helper_file_name());
    if adjacent.is_file() {
        return Ok(HelperSpec::dedicated(adjacent));
    }

    Err(format!(
        "native converter helper is not configured; package '{}' beside the host executable, set {HELPER_PATH_ENV} to its exact path, or register a self-hosting executable before conversion (refusing unsafe in-process fallback)",
        helper_file_name().to_string_lossy()
    ))
}

fn helper_file_name() -> OsString {
    #[cfg(windows)]
    {
        OsString::from(format!("{HELPER_FILE_STEM}.exe"))
    }
    #[cfg(not(windows))]
    {
        OsString::from(HELPER_FILE_STEM)
    }
}

fn invoke_helper(
    spec: &HelperSpec,
    request: Request<'_>,
    timeout: Duration,
) -> Result<String, String> {
    let request_bytes = encode_request(request)?;
    let mut command = Command::new(&spec.path);
    command
        .args(&spec.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| {
        format!(
            "failed to spawn native converter helper {}: {e}",
            spec.path.display()
        )
    })?;

    let pipes = (child.stdin.take(), child.stdout.take(), child.stderr.take());
    let (Some(mut stdin), Some(stdout), Some(stderr)) = pipes else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("converter helper pipes were not established".to_string());
    };

    // Keep all pipe I/O off the deadline/polling thread. A malicious or broken
    // helper that never reads stdin, floods output, or fills stderr therefore
    // cannot bypass the wall-clock timeout or force unbounded host allocation.
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        stdin.write_all(&request_bytes)?;
        stdin.flush()
    });
    let stdout_reader = std::thread::spawn(move || {
        read_limited(stdout, RESPONSE_HEADER_BYTES + MAX_CONVERTER_OUTPUT_BYTES)
    });
    let stderr_reader = std::thread::spawn(move || read_limited(stderr, MAX_STDERR_BYTES));

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = writer.join();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(format!(
                    "native converter helper timed out after {} ms and was terminated",
                    timeout.as_millis()
                ));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = writer.join();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(format!("failed to poll native converter helper: {e}"));
            }
        }
    };

    let write_result = writer
        .join()
        .map_err(|_| "converter helper stdin writer panicked".to_string())?;
    let stdout = stdout_reader
        .join()
        .map_err(|_| "converter helper stdout reader panicked".to_string())??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "converter helper stderr reader panicked".to_string())??;

    if !status.success() {
        return Err(helper_exit_error(status, &stderr));
    }
    write_result.map_err(|e| format!("failed to send request to converter helper: {e}"))?;
    decode_response(&stdout)
}

fn helper_exit_error(status: ExitStatus, stderr: &[u8]) -> String {
    let diagnostic = String::from_utf8_lossy(stderr);
    let diagnostic = diagnostic.trim();
    if diagnostic.is_empty() {
        format!("native converter helper crashed or exited unsuccessfully ({status})")
    } else {
        format!("native converter helper crashed or exited unsuccessfully ({status}): {diagnostic}")
    }
}

fn read_limited<R: Read>(reader: R, max: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    reader
        .take((max + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("failed reading converter helper pipe: {e}"))?;
    if bytes.len() > max {
        return Err(format!(
            "converter helper output exceeded the {max}-byte safety cap"
        ));
    }
    Ok(bytes)
}

fn encode_request(request: Request<'_>) -> Result<Vec<u8>, String> {
    let (op, flags, globals, body) = match request {
        Request::Probe => (OP_PROBE, 0, "", ""),
        Request::Convert {
            file_globals,
            body,
            optimize,
        } => {
            validate_input_size(file_globals, body)?;
            let mut flags = u32::from(optimize) * FLAG_OPTIMIZE;
            if file_globals.is_some() {
                flags |= FLAG_GLOBALS_PRESENT;
            }
            (OP_CONVERT, flags, file_globals.unwrap_or(""), body)
        }
    };
    let globals_len = u32::try_from(globals.len())
        .map_err(|_| "converter globals length exceeds protocol".to_string())?;
    let body_len = u32::try_from(body.len())
        .map_err(|_| "converter body length exceeds protocol".to_string())?;
    let mut bytes = Vec::with_capacity(REQUEST_HEADER_BYTES + globals.len() + body.len());
    bytes.extend_from_slice(&REQUEST_MAGIC);
    bytes.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    bytes.extend_from_slice(&op.to_le_bytes());
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&globals_len.to_le_bytes());
    bytes.extend_from_slice(&body_len.to_le_bytes());
    bytes.extend_from_slice(globals.as_bytes());
    bytes.extend_from_slice(body.as_bytes());
    Ok(bytes)
}

fn decode_request(bytes: &[u8]) -> Result<DecodedRequest, String> {
    if bytes.len() < REQUEST_HEADER_BYTES {
        return Err("converter helper request header is truncated".to_string());
    }
    if bytes[0..4] != REQUEST_MAGIC {
        return Err("converter helper request has bad magic".to_string());
    }
    let version = read_u32(bytes, 4)?;
    if version != PROTOCOL_VERSION {
        return Err(format!(
            "unsupported converter helper protocol version {version}"
        ));
    }
    let op = read_u32(bytes, 8)?;
    let flags = read_u32(bytes, 12)?;
    let globals_len = read_u32(bytes, 16)? as usize;
    let body_len = read_u32(bytes, 20)? as usize;
    let payload_len = globals_len
        .checked_add(body_len)
        .ok_or_else(|| "converter helper request length overflow".to_string())?;
    if payload_len > MAX_CONVERTER_INPUT_BYTES {
        return Err(format!(
            "converter helper request exceeds the {}-byte safety cap",
            MAX_CONVERTER_INPUT_BYTES
        ));
    }
    let expected = REQUEST_HEADER_BYTES
        .checked_add(payload_len)
        .ok_or_else(|| "converter helper request length overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "converter helper request length mismatch: expected {expected}, got {}",
            bytes.len()
        ));
    }
    if flags & !ALLOWED_FLAGS != 0 {
        return Err("converter helper request has unknown flags".to_string());
    }

    match op {
        OP_PROBE if flags == 0 && payload_len == 0 => Ok(DecodedRequest::Probe),
        OP_PROBE => Err("converter helper probe carries unexpected payload".to_string()),
        OP_CONVERT => {
            let globals_start = REQUEST_HEADER_BYTES;
            let body_start = globals_start + globals_len;
            let globals = std::str::from_utf8(&bytes[globals_start..body_start])
                .map_err(|_| "converter globals are not UTF-8".to_string())?;
            let body = std::str::from_utf8(&bytes[body_start..])
                .map_err(|_| "converter body is not UTF-8".to_string())?;
            let globals_present = flags & FLAG_GLOBALS_PRESENT != 0;
            if !globals_present && globals_len != 0 {
                return Err("converter globals payload lacks its presence flag".to_string());
            }
            Ok(DecodedRequest::Convert(DecodedConvertRequest {
                file_globals: globals_present.then(|| globals.to_owned()),
                body: body.to_owned(),
                optimize: flags & FLAG_OPTIMIZE != 0,
            }))
        }
        _ => Err(format!("unknown converter helper operation {op}")),
    }
}

fn encode_response(result: Result<String, String>) -> Vec<u8> {
    let (status, payload) = match result {
        Ok(payload) if payload.len() <= MAX_CONVERTER_OUTPUT_BYTES => (0u32, payload),
        Ok(payload) => (
            1u32,
            format!(
                "native converter output is {} bytes, exceeding the {}-byte safety cap",
                payload.len(),
                MAX_CONVERTER_OUTPUT_BYTES
            ),
        ),
        Err(error) => (1u32, truncate_utf8(error, MAX_CONVERTER_OUTPUT_BYTES)),
    };
    let payload_len = payload.len() as u32;
    let mut bytes = Vec::with_capacity(RESPONSE_HEADER_BYTES + payload.len());
    bytes.extend_from_slice(&RESPONSE_MAGIC);
    bytes.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    bytes.extend_from_slice(&status.to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(payload.as_bytes());
    bytes
}

fn decode_response(bytes: &[u8]) -> Result<String, String> {
    if bytes.len() < RESPONSE_HEADER_BYTES {
        return Err("native converter helper returned a truncated response".to_string());
    }
    if bytes[0..4] != RESPONSE_MAGIC {
        return Err("native converter helper returned bad response magic".to_string());
    }
    let version = read_u32(bytes, 4)?;
    if version != PROTOCOL_VERSION {
        return Err(format!(
            "native converter helper returned unsupported protocol version {version}"
        ));
    }
    let status = read_u32(bytes, 8)?;
    let payload_len = read_u32(bytes, 12)? as usize;
    if payload_len > MAX_CONVERTER_OUTPUT_BYTES
        || bytes.len() != RESPONSE_HEADER_BYTES + payload_len
    {
        return Err("native converter helper returned an invalid payload length".to_string());
    }
    let payload = std::str::from_utf8(&bytes[RESPONSE_HEADER_BYTES..])
        .map_err(|_| "native converter helper returned non-UTF-8 output".to_string())?
        .to_owned();
    match status {
        0 => Ok(payload),
        1 => Err(payload),
        other => Err(format!(
            "native converter helper returned unknown status {other}"
        )),
    }
}

fn truncate_utf8(mut value: String, max: usize) -> String {
    if value.len() <= max {
        return value;
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    value
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "converter helper protocol field is truncated".to_string())?;
    Ok(u32::from_le_bytes(
        raw.try_into().expect("length checked above"),
    ))
}

/// Execute exactly one framed request from stdin and emit exactly one framed
/// response to stdout. Only this helper-process path may call the legacy C++ ABI.
pub fn run_converter_helper_stdio() -> i32 {
    let mut request = Vec::new();
    let read_result = std::io::stdin()
        .take((REQUEST_HEADER_BYTES + MAX_CONVERTER_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut request);
    let result = match read_result {
        Ok(_) if request.len() <= REQUEST_HEADER_BYTES + MAX_CONVERTER_INPUT_BYTES => {
            match decode_request(&request) {
                Ok(DecodedRequest::Probe) => {
                    if crate::NATIVE_CONVERTER_AVAILABLE {
                        Ok("available".to_string())
                    } else {
                        Err("helper was built without the native converter payload".to_string())
                    }
                }
                Ok(DecodedRequest::Convert(request)) => crate::call_c_convert_in_process(
                    request.file_globals.as_deref(),
                    &request.body,
                    request.optimize,
                ),
                Err(error) => Err(error),
            }
        }
        Ok(_) => Err("converter helper request exceeded its input cap".to_string()),
        Err(error) => Err(format!("failed reading converter helper request: {error}")),
    };

    let response = encode_response(result);
    let mut stdout = std::io::stdout();
    match stdout.write_all(&response).and_then(|()| stdout.flush()) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("failed writing converter helper response: {error}");
            74
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_input_is_rejected_before_helper_discovery() {
        let body = "x".repeat(MAX_CONVERTER_INPUT_BYTES + 1);
        let error = convert(None, &body, true).unwrap_err();
        assert!(error.contains("safety cap"), "{error}");
    }

    #[test]
    fn protocol_error_response_is_fail_closed() {
        let bytes = encode_response(Err("compiler rejected shader".to_string()));
        assert_eq!(
            decode_response(&bytes),
            Err("compiler rejected shader".to_string())
        );
    }

    #[test]
    fn missing_helper_is_reported_as_spawn_failure() {
        let spec = HelperSpec::dedicated(PathBuf::from(
            "/definitely/not/a/particle-milkdrop-converter-helper",
        ));
        let error = invoke_helper(&spec, Request::Probe, Duration::from_millis(50)).unwrap_err();
        assert!(error.contains("failed to spawn"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn crashed_helper_is_detected() {
        let spec = HelperSpec {
            path: PathBuf::from("/bin/sh"),
            args: vec![OsString::from("-c"), OsString::from("kill -ABRT $$")],
        };
        let error = invoke_helper(&spec, Request::Probe, Duration::from_secs(1)).unwrap_err();
        assert!(
            error.contains("crashed or exited unsuccessfully"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn hung_helper_is_killed_at_deadline() {
        let spec = HelperSpec {
            path: PathBuf::from("/bin/sh"),
            args: vec![OsString::from("-c"), OsString::from("exec sleep 5")],
        };
        let started = Instant::now();
        let error = invoke_helper(&spec, Request::Probe, Duration::from_millis(75)).unwrap_err();
        assert!(error.contains("timed out"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn request_roundtrip_preserves_fields() {
        let bytes = encode_request(Request::Convert {
            file_globals: Some("float g = 1;"),
            body: "ret = float3(g, 0, 0);",
            optimize: true,
        })
        .unwrap();
        let DecodedRequest::Convert(decoded) = decode_request(&bytes).unwrap() else {
            panic!("expected conversion request")
        };
        assert_eq!(decoded.file_globals.as_deref(), Some("float g = 1;"));
        assert_eq!(decoded.body, "ret = float3(g, 0, 0);");
        assert!(decoded.optimize);
    }
}
