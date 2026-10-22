import type { PageLoad } from './$types';
import { initStores } from '$lib/stores/init';
import { validateApiKey } from '$lib/utils';

export const load: PageLoad = async ({ fetch }) => {
	await initStores();
	await validateApiKey(fetch);
};
