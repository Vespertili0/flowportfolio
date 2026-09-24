import {
    loadGitHubContext,
    validateEnv,
    fetchPrDiff,
    buildDiffContext,
    sanitise,
    spawnJulesSession,
    awaitSessionResult,
    postGitHubComment,
    postCommitStatus
} from './jules_utils.mjs';

async function run() {
    const { prNumber, headSha, headRef, prTitle, prBody } = loadGitHubContext();
    const { apiKey, githubToken, repo } = validateEnv();
    const baseBranch = process.env.GITHUB_BASE_REF || 'dev';

    try {
        console.log(`🚀 Fetching PR diff for ${repo} PR #${prNumber}...`);
        const diff = await fetchPrDiff(repo, prNumber, githubToken);

        console.log(`📝 Constructing targeted review prompt...`);
        const safeHeadRef = sanitise(headRef);
        const diffContext = buildDiffContext(repo, prTitle, prBody, diff);

        const reviewPrompt = `You are an expert code reviewer and release manager. Review the pull request below with high precision and minimal false positives.

# Version Bump Instruction
Before the diff context, assess whether this PR requires a Semantic Versioning bump (patch, minor, or major).
Check the current \`version\` in \`pyproject.toml\` on the feature branch (\`${safeHeadRef}\`) against \`pyproject.toml\` on \`${baseBranch}\`. If it is already bumped, do not perform any file modifications.
If a bump is required and not already done:
1. Canonicalise the version from \`pyproject.toml\`.
2. Update \`version\` in \`pyproject.toml\` to the new SemVer version.
3. Update \`__version__\` in \`flowportfolio/__init__.py\` to match.
4. Create a pull request targeting the \`${safeHeadRef}\` branch with title \`chore(release): bump version to X.Y.Z\`.

${diffContext}

# What to review
Focus ONLY on lines changed in this diff. Evaluate for:
- Correctness: logic errors, null/undefined handling, race conditions, edge cases.
- Security: injection risks, hardcoded secrets, auth flaws, sensitive data in logs.
- Reliability: missing error handling, unhandled promise rejections.
- Dependency Version Verification: when evaluating dependency versions, if you do not recognise high version numbers, flag them as a [WARN] stating your cut-off limitation, do not block them as hallucinations.

# Severity tags
Tag each finding EXACTLY one of:
- [BLOCKING] — high-confidence correctness/security flaws (>80% confidence).
- [WARN] — meaningful concerns worth addressing but not blocking.
- [NIT] — small readability or consistency notes (max 3).

# Output format (STRICT)
Respond in Markdown using sections: ## Summary, ## Strengths, ## Findings (grouped by severity heading), and ## SemVer Assessment. End with EXACTLY one line:
\`VERDICT: approve\` — no blocking issues.
\`VERDICT: comment\` — has warnings/nits but nothing blocking.
\`VERDICT: block\` — one or more BLOCKING issues.`;

        console.log(`⏳ Spawning Jules cloud review & autoPr session on branch ${safeHeadRef}...`);
        const session = await spawnJulesSession(apiKey, repo, safeHeadRef, reviewPrompt, { autoPr: true });

        console.log("✅ Jules session successfully dispatched to cloud environment.");

        console.log(`💬 Posting dispatch comment back to PR #${prNumber}...`);
        await postGitHubComment(
            repo,
            prNumber,
            githubToken,
            `## 🤖 Jules Engaged\n\nA Jules cloud session (\`${session.id}\`) has been dispatched to perform code review and assess SemVer bumping.\n\nIf a version bump is required, Jules will open a separate PR against this feature branch (\`${safeHeadRef}\`) with the updated files.`
        );

        console.log(`🚦 Updating commit status to pending...`);
        await postCommitStatus(
            repo,
            headSha,
            githubToken,
            'pending',
            'jules/review',
            'Jules review & autoPr session is in progress'
        );

        let reviewMarkdown = "";
        try {
            reviewMarkdown = await awaitSessionResult(session);
        } catch (streamError) {
            // Log full error details to Actions console only — never expose to public PR comments
            console.error("⚠️ Stream failed or timed out:", streamError.stack || streamError.message || String(streamError));
            const softTimeoutComment = `⚠️ **Jules review stream interrupted.**\nThe session (\`${session.id}\`) was dispatched and may still complete the version bump PR in the background, but the review feedback stream timed out or did not report back.\n\nPlease check the GitHub Actions workflow logs for details.`;

            await postGitHubComment(repo, prNumber, githubToken, softTimeoutComment);
            await postCommitStatus(repo, headSha, githubToken, 'error', 'jules/review', 'Review stream timed out or failed');
            console.log("❌ Workflow failed due to stream timeout.");
            process.exit(1);
        }

        console.log("✅ Jules Review Completed successfully!");

        console.log(`💬 Posting final review comment back to PR #${prNumber}...`);
        await postGitHubComment(repo, prNumber, githubToken, `## 🤖 Jules Review\n\n${reviewMarkdown}\n\n---\n_Session: \`${session.id}\`_`);

        const isBlocked = reviewMarkdown.toUpperCase().includes('VERDICT: BLOCK') || reviewMarkdown.includes('[BLOCKING]');
        console.log(`🚦 Updating final commit status for SHA ${headSha}...`);
        await postCommitStatus(
            repo,
            headSha,
            githubToken,
            isBlocked ? 'failure' : 'success',
            'jules/review',
            isBlocked ? 'Blocking issues found by Jules' : 'Review complete (verdict: approve)'
        );

        console.log("✅ Workflow complete!");
        process.exit(0);

    } catch (error) {
        // Log full error details to Actions console only — never expose to public PR comments
        console.error("❌ Error running Jules PR Review:", error.stack || error.message || String(error));
        try {
            await postGitHubComment(
                repo,
                prNumber,
                githubToken,
                `⚠️ **Jules PR review failed to complete.**\n\nPlease check the GitHub Actions workflow logs for more details.`
            );
        } catch (commentError) {
            console.error("❌ Failed to post fallback error comment:", commentError.stack || commentError.message || String(commentError));
        }
        process.exit(1);
    }
}

run();
