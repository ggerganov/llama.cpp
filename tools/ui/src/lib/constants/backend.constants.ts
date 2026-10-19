/** Prefix for generated ids of user-added backends. */
export const BACKEND_ID_PREFIX = 'backend';

/** Chat completions path every backend serves. */
export const DEFAULT_BACKEND_CHAT_PATH = '/v1/chat/completions';

/** Models listing path every backend serves. */
export const DEFAULT_BACKEND_MODELS_PATH = '/v1/models';

/** Id of the built-in backend that points at the server serving this UI. */
export const LOCAL_BACKEND_ID = 'local';

/** Favicon extract keyed by domain, for backends with no bundled mark. */
export const FAVICON_SERVICE_URL = 'https://www.google.com/s2/favicons?domain=';
