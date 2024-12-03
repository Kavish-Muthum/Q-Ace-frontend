import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import icon from "astro-icon";
import node from '@astrojs/node';

// https://astro.build/config
export default defineConfig({
  site: "https://astroship.web3templates.com",
  integrations: [tailwind(), mdx(), sitemap(), icon()],
  output: 'server',
  adapter: node({
      mode: 'standalone'
  }),
  vite: {
    define: {
        'process.env.PERPLEXITY_API_KEY': JSON.stringify(process.env.PERPLEXITY_API_KEY),
        'process.env.OPENAI_API_KEY': JSON.stringify(process.env.OPENAI_API_KEY)
    }
  }
});
