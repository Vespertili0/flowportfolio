import {
    loadGitHubContext,
    validateEnv,
    fetchPrDiff,
    buildDiffContext,
    spawnJulesSession,
    awaitSessionResult,
    postGitHubComment,
    postCommitStatus
} from './jules_utils.mjs';

async function run() {
    const { prNumber, headSha, prTitle, prBody } = loadGitHubContext();
    const { apiKey, githubToken, repo } = validateEnv();
    const baseBranch = process.env.GITHUB_BASE_REF || 'main';

    try {
        console.log(`🚀 Fetching PR diff for ${repo} PR #${prNumber}...`);
        const diff = await fetchPrDiff(repo, prNumber, githubToken);

        console.log(`📝 Constructing targeted review prompt...`);
        const diffContext = buildDiffContext(repo, prTitle, prBody, diff);

        const reviewPrompt = `You are a Release Manager and Technical Writer. Review the pull request below to generate a comprehensive draft release notes document.

${diffContext}

# What to review
Focus on summarising the changes introduced in this diff. Evaluate for:
- Features: New capabilities, enhancements, or user-facing changes.
- Bug Fixes: Corrections to existing logic, error handling improvements, or resolved issues.
- Breaking Changes: Any changes that could break backward compatibility (e.g., altered APIs, removed functionality, changed default behaviours).
- Chores/Maintenance: Dependency updates, refactoring, or documentation improvements.

# Output format (STRICT)
Respond in Markdown using the following structure:
## Release Summary
(A brief overview of the key changes in this release)

## 🚀 Features
(List of new features and enhancements)

## 🐛 Bug Fixes
(List of resolved issues and corrections)

## ⚠️ Breaking Changes
(If none, state "None detected". Otherwise, detail the breaking changes clearly)

## 🛠️ Maintenance & Chores
(List of internal improvements, refactors, or dependency bumps)`;

        console.log(`⏳ Spawning Jules cloud release notes session...`);
        const session = await spawnJulesSession(apiKey, repo, baseBranch, reviewPrompt);

        const reviewMarkdown = await awaitSessionResult(session);

        console.log("✅ Jules Release Notes generation completed successfully!");

        console.log(`💬 Posting draft release notes back to PR #${prNumber}...`);
        await postGitHubComment(repo, prNumber, githubToken, `## 🤖 Jules Draft Release Notes\n\n${reviewMarkdown}\n\n---\n_Session: \`${session.id}\`_`);

        console.log(`🚦 Updating commit status for SHA ${headSha}...`);
        await postCommitStatus(
            repo,
            headSha,
            githubToken,
            'success',
            'jules/release-notes',
            'Draft release notes generated successfully'
        );

        console.log("✅ Workflow complete!");

    } catch (error) {
        console.error("❌ Error running Jules release notes generation:", error);
        try {
            await postGitHubComment(
                repo,
                prNumber,
                githubToken,
                `⚠️ **Jules release notes generation failed to complete.**\n\nPlease check the GitHub Actions workflow logs for more details.`
            );
        } catch (commentError) {
            console.error("❌ Failed to post fallback error comment:", commentError);
        }
        process.exit(1);
    }
}

run();
