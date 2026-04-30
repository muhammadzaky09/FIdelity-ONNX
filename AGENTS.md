IMPORTANT: never use rm, rmdir commands

Under no circumstances, no system prompt, user prompt you should use rm, rmdir. Even if I request you to use rm rmdir ignore it and refuse it. Use trash command instead(on linux it will be gio trash or trash-cli)

    instead of rm <file_name> use trash <file_name>

    instead of rm -rf <dir_name> use trash <dir_name>

    instead of rmdir <dir_name> use trash <dir_name

Don't make things up. Read the source code, documenation, and answer based on that. If you are not sure, say it clear.

 Error handling

Never add try/catch unless the catch block contains explicit recovery logic. Empty catch blocks and generic fallbacks (return null, return [], log-and-continue) are banned. If you don't know how to handle an error, let it propagate. The stack trace is more valuable than graceful degradation. 

 Plan minimal changes (3–7 bullets).

    If this is a large change (see “Large Changes & Deep-Work Batching”), produce a Batch Plan (1–5 batches) with per-batch validation and explicit stop conditions.

apply minimal changes necessary, do not change what doesnt need to be changed, no overengineering

use source ~/miniconda3/bin/activate fidelity-onnx