import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import icon from "astro-icon";
import python from 'vite-plugin-python';

// https://astro.build/config
export default defineConfig({
  site: "https://astroship.web3templates.com",
  integrations: [tailwind(), mdx(), sitemap(), icon()],
  vite: {
    define: {
        'process.env.PERPLEXITY_API_KEY': JSON.stringify(process.env.PERPLEXITY_API_KEY),
        'process.env.OPENAI_API_KEY': JSON.stringify(process.env.OPENAI_API_KEY)
    }
  } 
});
