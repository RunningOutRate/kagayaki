# kagayaki
レイトレ合宿9で提出したレンダラです．開発中であり，現在では正しい動作の保証はできません．  
主な機能  
・BVH  
・NextEventEstimationとMultipleImportanceSampling  
・Texturing  
・環境テクスチャ  
・LambertDiffuse  
・DisneyBRDF（不安定）  
・SVGF（不安定）  
・objのロード(tinyobjloader)  
・pngでのレンダリング結果保存(stb)  
・glTFのロードには現在対応しておりません

開発環境  
・C++23  
・CUDA12.1　

 追記(2024/01/05)  
・この実装には怪しいところや苦しいところがたくさんあって手が付けられないため，再実装することにしております（このリポジトリは更新しません）．
