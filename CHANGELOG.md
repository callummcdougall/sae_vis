# CHANGELOG



## v0.2.18 (2024-04-22)

### Fix

* fix: remove circular dep with sae lens ([`1dd9f6c`](https://github.com/callummcdougall/sae_vis/commit/1dd9f6cd22f879e8d6904ba72f3e52b4344433cd))

### Unknown

* Merge pull request #43 from callummcdougall/move_saelens_dep

Remove dependency on saelens from pyproject, add to demo.ipynb ([`147d87e`](https://github.com/callummcdougall/sae_vis/commit/147d87ee9534d30e764851cbe73aadb5783d2515))

* Add missing matplotlib ([`572a3cc`](https://github.com/callummcdougall/sae_vis/commit/572a3cc79709a14117bbeafb871a33f0107600d8))

* Remove dependency on saelens from pyproject, add to demo.ipynb ([`1e6f3cf`](https://github.com/callummcdougall/sae_vis/commit/1e6f3cf9b2bcfb381a73d9333581c430faa531fd))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`4e7a24c`](https://github.com/callummcdougall/sae_vis/commit/4e7a24c37444f11d718035eede68ac728d949a20))

* fix conflicts ([`ea3d624`](https://github.com/callummcdougall/sae_vis/commit/ea3d624013b9aa7cbd2d6eaa7212a1f7c4ee8e28))

* Merge pull request #41 from callummcdougall/allow_disable_buffer

oops I forgot to switch back to main before pushing ([`1312cd0`](https://github.com/callummcdougall/sae_vis/commit/1312cd09d6e274b1163e79d2ac01f2df54c65157))

* Merge branch &#39;main&#39; into allow_disable_buffer ([`e7edf5a`](https://github.com/callummcdougall/sae_vis/commit/e7edf5a9bae4714bf4983ce6a19a0fe6fdf1f118))

* 16 ([`64e7018`](https://github.com/callummcdougall/sae_vis/commit/64e701849570d9e172dc065812c9a3e7149a9176))


## v0.2.17 (2024-04-21)

### Chore

* chore: setting up semantic-release ([`09075af`](https://github.com/callummcdougall/sae_vis/commit/09075afbec279fb89d157f73e9a0ed47ba66d3c8))

### Unknown

* Merge pull request #40 from chanind/semantic-release-autodeploy

chore: setting up semantic-release for auto-deploy ([`a4d44d1`](https://github.com/callummcdougall/sae_vis/commit/a4d44d1a0e86055fb82ef41f51f0adbb7868df3c))

* version 0.2.16 ([`afca0be`](https://github.com/callummcdougall/sae_vis/commit/afca0be8826e0c007b5730fa9fa18454699d16a3))

* Merge pull request #38 from chanind/type-checking

Enabling type checking with Pyright ([`f1fd792`](https://github.com/callummcdougall/sae_vis/commit/f1fd7926f46f00dca46024377f53aa8f2db98773))

* Merge pull request #39 from callummcdougall/fix_loading_saelens_sae

FIX: SAELens new format has &#34;scaling_factor&#34; key, which causes assert to fail ([`983aee5`](https://github.com/callummcdougall/sae_vis/commit/983aee562aea31e90657caf8c6ab6e450e952120))

* Fix Formatting ([`13b8106`](https://github.com/callummcdougall/sae_vis/commit/13b81062485f5dce2568e7832bfb2aae218dd4e9))

* Merge branch &#39;main&#39; into fix_loading_saelens_sae ([`21b0086`](https://github.com/callummcdougall/sae_vis/commit/21b0086b8af3603441795e925a15e7cded122acb))

* Allow SAELens autoencoder keys to be superset of required keys, instead of exact match ([`6852170`](https://github.com/callummcdougall/sae_vis/commit/6852170d55e7d3cf22632c5807cfab219516da98))

* enabling type checking with Pyright ([`05d14ea`](https://github.com/callummcdougall/sae_vis/commit/05d14eafea707d3db81e78b4be87199087cb8e37))

* Fix version ([`5a43916`](https://github.com/callummcdougall/sae_vis/commit/5a43916cbd9836396f051f7a258fdca8664e05e9))

* format ([`8f1506b`](https://github.com/callummcdougall/sae_vis/commit/8f1506b6eb7dc0a2d4437d2aa23a0898c46a156d))

* v0.2.17 ([`2bb14da`](https://github.com/callummcdougall/sae_vis/commit/2bb14daa88a0af601e13f4e51b50a2b00cd75b48))

* Use main branch of SAELens ([`2b34505`](https://github.com/callummcdougall/sae_vis/commit/2b345052bdc92ee9c1255cab0978916307a0a9dc))

* Update version 0.2.16 ([`bf90293`](https://github.com/callummcdougall/sae_vis/commit/bf902930844db9b0f8db4fbe8b3610557352660b))

* Merge pull request #36 from callummcdougall/allow_disable_buffer

FEATURE: Allow setting buffer to None, which gives the whole activation sequence ([`f5f9594`](https://github.com/callummcdougall/sae_vis/commit/f5f9594fcaf5edb6036a85446e092278004ea200))

* fix all indices view ([`5f87d52`](https://github.com/callummcdougall/sae_vis/commit/5f87d52154d6a8e8c8984836bbe8f85ee25f279d))

* Merge pull request #35 from callummcdougall/fix_gpt2_demo

Fix usage of SAELens and demo notebook ([`88b5933`](https://github.com/callummcdougall/sae_vis/commit/88b59338d3cadbd5c70f0c1117dff00f01a54e6a))

* Merge branch &#39;fix_gpt2_demo&#39; into allow_disable_buffer ([`ea57bfc`](https://github.com/callummcdougall/sae_vis/commit/ea57bfc2ee1e23666810982abf32e6e9cbb74193))

* Import updated SAELens, use correct tokens, fix missing file cfg.json file error. ([`14ba9b0`](https://github.com/callummcdougall/sae_vis/commit/14ba9b03d4ce791ba8f4cac553fb82a93c47dfb8))

* Merge pull request #34 from ArthurConmy/patch-1

Update README.md ([`3faac82`](https://github.com/callummcdougall/sae_vis/commit/3faac82686f546800492d8aeb5e1d5919cbf1517))

* Update README.md ([`416eca8`](https://github.com/callummcdougall/sae_vis/commit/416eca8073c6cb2b120c759330ec47f52ab32d1e))

* Merge pull request #33 from chanind/setup-poetry-and-ruff

Setting up poetry / ruff / github actions ([`287f30f`](https://github.com/callummcdougall/sae_vis/commit/287f30f1d8fc39ab583f202c9277e07e5eeeaf62))

* setting up poetry and ruff for linting/formatting ([`0e0eba9`](https://github.com/callummcdougall/sae_vis/commit/0e0eba9e4d54c746cddc835ef4f6ddf2bab96844))

* fix feature vis demo gpt ([`821781e`](https://github.com/callummcdougall/sae_vis/commit/821781e96b732a5909d8735714482c965891b2ea))

* Allow disabling the buffer ([`c1be9f8`](https://github.com/callummcdougall/sae_vis/commit/c1be9f8e4b8ee6d8f18c4a1a0445840304440c1d))

* add scatter plot support ([`6eab28b`](https://github.com/callummcdougall/sae_vis/commit/6eab28bef9ef5cd9360fef73e02763301fa1a028))

* update setup ([`8d2ca53`](https://github.com/callummcdougall/sae_vis/commit/8d2ca53e8a6bba860fe71368741d06a718adaa27))

* fix setup ([`9cae8f4`](https://github.com/callummcdougall/sae_vis/commit/9cae8f461bd780e23eb2d994f56b495ede16201a))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`ed8f8cb`](https://github.com/callummcdougall/sae_vis/commit/ed8f8cb7ad1fba2383dcdd471c33ce4a1b9f32e3))

* fix sae bug ([`247d14b`](https://github.com/callummcdougall/sae_vis/commit/247d14b55f209ed9ccf50e5ce091ed66ffbf19d2))

* Merge pull request #27 from wllgrnt/will-add-eindex-dependency

Update setup.py with eindex dependency ([`8d7ed12`](https://github.com/callummcdougall/sae_vis/commit/8d7ed123505ac7ecf93dd310f57888547aead1d7))

* Merge pull request #32 from hijohnnylin/pin_older_sae_training

Demo notebook errors under &#34;Multi-layer models&#34; vis ([`9ac1dac`](https://github.com/callummcdougall/sae_vis/commit/9ac1dac51af32909666977cb5b3794965c70f62f))

* Pin older commit of mats_sae_training ([`8ca7ac1`](https://github.com/callummcdougall/sae_vis/commit/8ca7ac14b919fedb91240630ac7072cac40a6d6a))

* two more deps ([`7f231a8`](https://github.com/callummcdougall/sae_vis/commit/7f231a83acfef2494c1866249f57e10c21a1a443))

* Update setup.py with eindex

Without this, &#39;pip install sae-vis&#39; will cause errors when e.g. you do &#39;from sae_vis.data_fetching_fns import get_feature_data&#39; ([`a9d7de9`](https://github.com/callummcdougall/sae_vis/commit/a9d7de90b492f7305758e15303ba890fb9b503d0))

* update version number ([`72e584b`](https://github.com/callummcdougall/sae_vis/commit/72e584b6492ed1ef3989968f6588a17fca758650))

* add gifs to readme ([`1393740`](https://github.com/callummcdougall/sae_vis/commit/13937405da31cca70cd1027aaca6c9cc84797ff1))

* test gif ([`4fbafa6`](https://github.com/callummcdougall/sae_vis/commit/4fbafa69343dc58dc18d0f78e393b5fcc9e24c0c))

* fix height issue ([`3f272f6`](https://github.com/callummcdougall/sae_vis/commit/3f272f61a954effef7bd648cc8117346da3bb971))

* fix pypi ([`7151164`](https://github.com/callummcdougall/sae_vis/commit/7151164cc0df8af278617147f07cbfbe3977cfeb))

* update setup ([`8c43478`](https://github.com/callummcdougall/sae_vis/commit/8c43478ad2eba8d3d4106fe4239c1229b8720fe6))

* Merge pull request #26 from hijohnnylin/update_html_anomalies

Update and add some HTML_ANOMALIES ([`1874a47`](https://github.com/callummcdougall/sae_vis/commit/1874a47a099ce32795bdbb5f98b9167dcca85ff2))

* Update and add some HTML_ANOMALIES ([`c541b7f`](https://github.com/callummcdougall/sae_vis/commit/c541b7f06108046ad1e2eb82c89f30f061f4411e))

* 0.2.9 ([`a5c8a6d`](https://github.com/callummcdougall/sae_vis/commit/a5c8a6d2008b818db90566cba50211845c753444))

* fix readme ([`5a8a7e3`](https://github.com/callummcdougall/sae_vis/commit/5a8a7e3173fc50fdb5ff0e56d7fa83e475af38a3))

* include feature tables ([`7c4c263`](https://github.com/callummcdougall/sae_vis/commit/7c4c263a2e069482d341b6265015664792bde817))

* add license ([`fa02a3d`](https://github.com/callummcdougall/sae_vis/commit/fa02a3dc93b721322b3902e2ac416ed156bf9d80))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`ca5efcd`](https://github.com/callummcdougall/sae_vis/commit/ca5efcdc81074d3c3002bd997b35e326a44a4a25))

* re-fix html anomalies ([`2fbae4c`](https://github.com/callummcdougall/sae_vis/commit/2fbae4c9a7dd663737bae25e73e978d40c59064a))

* Merge pull request #24 from chanind/fix-pypi-repo-link

fixing repo URL in setup.py ([`14a0be5`](https://github.com/callummcdougall/sae_vis/commit/14a0be54a57b1bc73ac4741611f9c8d1bd229e6f))

* fixing repo URL in setup.py ([`4faeca5`](https://github.com/callummcdougall/sae_vis/commit/4faeca5da06c0bb4384e202a91d895a217365d30))

* fix hook point bug ([`9b573b2`](https://github.com/callummcdougall/sae_vis/commit/9b573b27590db1cbd6c8ef08fca7ff8c9d26b340))

* Merge pull request #20 from chanind/fix-final-resid-layer

fixing bug if hook_point == hook_point_resid_final ([`d6882e3`](https://github.com/callummcdougall/sae_vis/commit/d6882e3f813ef0d399e07548871f61b1f6a98ac6))

* fixing bug using hook_point_resid_final ([`cfe9b30`](https://github.com/callummcdougall/sae_vis/commit/cfe9b3042cfe127d5f7958064ffe817c25a19b56))

* fix indexing speed ([`865ff64`](https://github.com/callummcdougall/sae_vis/commit/865ff64329538641cd863dc7668dfc77907fb384))

* enable JSON saving ([`feea47a`](https://github.com/callummcdougall/sae_vis/commit/feea47a342d52296b72784ed18ea628848d4c7d4))

* Merge pull request #19 from chanind/support-mlp-and-attn-out

supporting mlp and attn out hooks ([`1c5463b`](https://github.com/callummcdougall/sae_vis/commit/1c5463b12f85cd0598b4e2fba5c556b1e9c0fbbe))

* supporting mlp and attn out hooks ([`a100e58`](https://github.com/callummcdougall/sae_vis/commit/a100e586498e8cae14df475bc7924cdecaed71ea))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`083aeba`](https://github.com/callummcdougall/sae_vis/commit/083aeba0e4048d9976ec5cbee8df7dc8fd4db4e9))

* fix variable naming ([`2507918`](https://github.com/callummcdougall/sae_vis/commit/25079186b3f31d2271b1ecdb11f26904af7146d2))

* Merge pull request #18 from chanind/remove-build-artifacts

removing Python build artifacts and adding to .gitignore ([`b0e0594`](https://github.com/callummcdougall/sae_vis/commit/b0e0594590b4472b34052c6eb3ebceb6c9f58a11))

* removing Python build artifacts and adding to .gitignore ([`b6486f5`](https://github.com/callummcdougall/sae_vis/commit/b6486f56bea9d4bb7544c36afe70e6f891101b63))

* update readme ([`0ee3608`](https://github.com/callummcdougall/sae_vis/commit/0ee3608af396a1a6586dfb809f2f6480bb4f6390))

* update readme ([`f8351f8`](https://github.com/callummcdougall/sae_vis/commit/f8351f88e8432ccd4b2206e859daea316304d6c6))

* update version number ([`1e74408`](https://github.com/callummcdougall/sae_vis/commit/1e7440883f44a92705299430215f802fea4e1915))

* fix formatting and docstrings ([`b9fe2bb`](https://github.com/callummcdougall/sae_vis/commit/b9fe2bbb15a48e4b0415f6f4240d895990d54c9a))

* Merge pull request #17 from jordansauce/sae-agnostic-functions-new

Added SAE class agnostic functions ([`0039c6f`](https://github.com/callummcdougall/sae_vis/commit/0039c6f8f99d6e8a1b2ff56aa85f60a3eba3afb0))

* add to pypi ([`02a5b9a`](https://github.com/callummcdougall/sae_vis/commit/02a5b9acd15433cc59d438271b9bd5e12d62b662))

* Added sae class agnostic functions

Added parse_feature_data() and parse_prompt_data() ([`e2709d0`](https://github.com/callummcdougall/sae_vis/commit/e2709d0b4c55d73d6026f3b9ce534f59ce61f344))

* update notebook images ([`b87ad4d`](https://github.com/callummcdougall/sae_vis/commit/b87ad4d256f12c23605b0e7db307ee56913c93ef))

* fix layer parse and custom device ([`14c7ae9`](https://github.com/callummcdougall/sae_vis/commit/14c7ae9d0c8b7dad21b953cfc93fe7f34c74e149))

* update dropdown styling ([`83be219`](https://github.com/callummcdougall/sae_vis/commit/83be219bfe31b985a26762e06345c574aa0e6fe1))

* add custom prompt vis ([`cabdc5c`](https://github.com/callummcdougall/sae_vis/commit/cabdc5cb31f881cddf236490c41332c525d2ee74))

* d3 &amp; multifeature refactor ([`f79a919`](https://github.com/callummcdougall/sae_vis/commit/f79a919691862f60a9e30fe0f79fd8e771bc932a))

* remove readme links ([`4bcef48`](https://github.com/callummcdougall/sae_vis/commit/4bcef489b644dd3357b1975f3245d534f6f0d2e0))

* add demo html ([`629c713`](https://github.com/callummcdougall/sae_vis/commit/629c713345407562dc4ccd9875bf3cfab5480bdd))

* remove demos ([`beedea9`](https://github.com/callummcdougall/sae_vis/commit/beedea9667761534a5293015aff9cc17638666a5))

* fix quantile error ([`3a23cfd`](https://github.com/callummcdougall/sae_vis/commit/3a23cfd56f21fe0775a1a9957db340d15f75f51a))

* width 425 ([`f25c776`](https://github.com/callummcdougall/sae_vis/commit/f25c776d5cb746916d3f2fdf368cbd5448742949))

* fix device bug ([`85dfa49`](https://github.com/callummcdougall/sae_vis/commit/85dfa497bc804945911e80607ac31cf3afbdc759))

* dont return vocab dict ([`b4c7138`](https://github.com/callummcdougall/sae_vis/commit/b4c713873870acb4035986cc5bff3a4ce1e466c9))

* save as JSON, fix device ([`eba2cff`](https://github.com/callummcdougall/sae_vis/commit/eba2cff3eb6215558577a6b4d4f8cc716766b927))

* simple fixed and issues ([`b28a0f7`](https://github.com/callummcdougall/sae_vis/commit/b28a0f7c7e936f4bea05528d952dfcd438533cce))

* Merge pull request #8 from lucyfarnik/topk-empty-mask

Topk error handling for empty masks ([`2740c00`](https://github.com/callummcdougall/sae_vis/commit/2740c0047e78df7e56d7bcf707c909ac18e71c1f))

* Topk error handling for empty masks ([`1c2627e`](https://github.com/callummcdougall/sae_vis/commit/1c2627e237f8f67725fc44e60a190bc141d36fc8))

* viz to vis ([`216d02b`](https://github.com/callummcdougall/sae_vis/commit/216d02b550d6fbcb9b37d39c1b272a7dda91aadc))

* update readme links ([`f9b3f95`](https://github.com/callummcdougall/sae_vis/commit/f9b3f95e31e7150024be27ec62246f43bf9bcbb8))

* update for TL ([`1941db1`](https://github.com/callummcdougall/sae_vis/commit/1941db1e22093d6fc88fb3fcd6f4c7d535d8b3b4))

* Merge pull request #5 from lucyfarnik/transformer-lens-models

Compatibility with TransformerLens models ([`8d59c6c`](https://github.com/callummcdougall/sae_vis/commit/8d59c6c5a5f2b98c486e5c74130371ad9254d1c9))

* Merge branch &#39;main&#39; into transformer-lens-models ([`73057d7`](https://github.com/callummcdougall/sae_vis/commit/73057d7e2a3e4e9669fc0556e64190811ac8b52d))

* Merge pull request #4 from lucyfarnik/resid-saes-support

Added support for residual-adjacent SAEs ([`b02e98b`](https://github.com/callummcdougall/sae_vis/commit/b02e98b3b852c0613a890f8949d04b5560fb6fd6))

* Merge pull request #7 from lucyfarnik/fix-histogram-div-zero

Fixed division by zero in histogram calculation ([`3aee20e`](https://github.com/callummcdougall/sae_vis/commit/3aee20ea7f99cc07e6c5085fddb70cadd8327f4d))

* Merge pull request #6 from lucyfarnik/handling-dead-features

Edge case handling for dead features ([`9e43c30`](https://github.com/callummcdougall/sae_vis/commit/9e43c308e58769828234e1505f1c1102ba651dfd))

* add features argument ([`f24ef7e`](https://github.com/callummcdougall/sae_vis/commit/f24ef7ebebb3d4fd92e299858dbd5b968b78c69e))

* fix image link ([`22c8734`](https://github.com/callummcdougall/sae_vis/commit/22c873434dfa84e3aed5ee0aab0fd25b288428a6))

* Merge pull request #1 from lucyfarnik/read-me-links-fix

Fixed readme links pointing to the old colab ([`86f8e20`](https://github.com/callummcdougall/sae_vis/commit/86f8e2012e376b6c498e5e708324f812af6fbc98))

* Fixed division by zero in histogram calculation ([`e986e90`](https://github.com/callummcdougall/sae_vis/commit/e986e907cc42790efc93ce75ebf7b28a0278aaa2))

* Added readme section about models ([`7523e7f`](https://github.com/callummcdougall/sae_vis/commit/7523e7f6363e030196496b3c6a3dc70b234c2d9a))

* Fixed readme links pointing to the old colab ([`28ef1cb`](https://github.com/callummcdougall/sae_vis/commit/28ef1cbd1b91f6c09c842f48e1f997d189ca04e7))

* Edge case handling for dead features ([`5197aee`](https://github.com/callummcdougall/sae_vis/commit/5197aee2c9f92bce7c5fd6d22201152a68c2e6ca))

* Compatibility with TransformerLens models ([`ba708e9`](https://github.com/callummcdougall/sae_vis/commit/ba708e987be6cc7a09d34ea8fb83de009312684d))

* Added support for MPS ([`196c0a2`](https://github.com/callummcdougall/sae_vis/commit/196c0a24d0e8277b327eb2d57662075f9106990b))

* Added support for residual-adjacent SAEs ([`89aacf1`](https://github.com/callummcdougall/sae_vis/commit/89aacf1b22aa81b393b10eca8611c9dbf406c638))

* black font ([`d81e74d`](https://github.com/callummcdougall/sae_vis/commit/d81e74d575326ef786881fb9182a768f9de2cb70))

* fix html bug ([`265dedd`](https://github.com/callummcdougall/sae_vis/commit/265dedd376991230e2041fd37d5b6a0eda048545))

* add jax and dataset deps ([`f1caeaf`](https://github.com/callummcdougall/sae_vis/commit/f1caeafc9613e27c7663447cf862301ac11d842d))

* remove TL dependency ([`155991f`](https://github.com/callummcdougall/sae_vis/commit/155991fe61d0199d081d344ac44996edce35d118))

* first commit ([`7782eb6`](https://github.com/callummcdougall/sae_vis/commit/7782eb6d5058372630c5bbb8693eb540a7bceaf4))
