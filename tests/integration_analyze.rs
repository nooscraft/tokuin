/// Integration tests for analyze-prompts command.
use std::fs;
use tempfile::TempDir;

fn create_test_prompt_library() -> TempDir {
    let dir = TempDir::new().unwrap();
    let base = dir.path();

    // Create various prompt files
    fs::write(base.join("simple.txt"), "Hello, world!").unwrap();
    fs::write(
        base.join("medium.txt"),
        "This is a medium length prompt that contains several sentences and should result in a reasonable token count.",
    )
    .unwrap();
    fs::write(
        base.join("long.txt"),
        "This is a very long prompt that contains many sentences and should result in a high token count. ".repeat(50),
    )
    .unwrap();

    // Create JSON prompt
    fs::write(
        base.join("chat.json"),
        r#"[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]"#,
    )
    .unwrap();

    // Create duplicate files
    fs::write(base.join("dup1.txt"), "Duplicate content").unwrap();
    fs::write(base.join("dup2.txt"), "Duplicate content").unwrap();

    // Create subdirectory
    let subdir = base.join("subdir");
    fs::create_dir(&subdir).unwrap();
    fs::write(subdir.join("nested.txt"), "Nested prompt").unwrap();

    dir
}

#[test]
fn test_analyze_folder_basic() {
    let test_dir = create_test_prompt_library();
    let output = std::process::Command::new("cargo")
        .args(&[
            "run",
            "--",
            "analyze-prompts",
            test_dir.path().to_str().unwrap(),
            "--model",
            "gpt-4",
            "--format",
            "json",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "Command should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("\"total_prompts\""),
        "Should contain total_prompts"
    );
    assert!(
        stdout.contains("\"token_distribution\""),
        "Should contain distribution"
    );
}

#[test]
fn test_analyze_folder_text_output() {
    let test_dir = create_test_prompt_library();
    let output = std::process::Command::new("cargo")
        .args(&[
            "run",
            "--",
            "analyze-prompts",
            test_dir.path().to_str().unwrap(),
            "--model",
            "gpt-4",
            "--format",
            "text",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "Command should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Prompt Library Insights"),
        "Should contain header"
    );
    assert!(stdout.contains("Total Prompts"), "Should contain summary");
}

#[test]
fn test_analyze_folder_with_context_limit() {
    let test_dir = create_test_prompt_library();
    let output = std::process::Command::new("cargo")
        .args(&[
            "run",
            "--",
            "analyze-prompts",
            test_dir.path().to_str().unwrap(),
            "--model",
            "gpt-4",
            "--context-limit",
            "100",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "Command should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Should detect prompts exceeding limit
    assert!(
        stdout.contains("exceeds") || stdout.contains("Exceeding"),
        "Should detect exceeded limits"
    );
}

#[test]
fn test_analyze_folder_nonexistent_dir() {
    let output = std::process::Command::new("cargo")
        .args(&[
            "run",
            "--",
            "analyze-prompts",
            "/nonexistent/directory",
            "--model",
            "gpt-4",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success(), "Command should fail");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("does not exist") || stderr.contains("Directory"),
        "Should show error message"
    );
}

#[test]
fn test_analyze_folder_top_n() {
    let test_dir = create_test_prompt_library();
    let output = std::process::Command::new("cargo")
        .args(&[
            "run",
            "--",
            "analyze-prompts",
            test_dir.path().to_str().unwrap(),
            "--model",
            "gpt-4",
            "--top-n",
            "3",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "Command should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Should show top 3 most expensive
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("tokens") && l.contains("invocation"))
        .collect();
    assert!(lines.len() <= 3, "Should show at most 3 top prompts");
}
