
name = "testBar2";

model = createpde('structural','modal-solid');

model.importGeometry(name  + ".stl");

ret = model.generateMesh('GeometricOrder','linear','Hmin',0.4)
pdeplot3D(model.Mesh)

nodes = model.Mesh.Nodes;
elems = model.Mesh.Elements;


%%


f = fopen(name + ".txt",'w');
fprintf(f,"%d %d\n", size(nodes,2), size(elems,2));
fprintf(f,"%27.16g %27.16g %27.16g \n", nodes);
fprintf(f,"%d %d %d %d\n", elems);



fclose(f);

%% test


