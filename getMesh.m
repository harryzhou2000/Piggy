
name = "testBar1";

model = createpde('structural','modal-solid');

model.importGeometry(name  + ".stl");


ret = model.generateMesh('GeometricOrder','linear','Hmin',0.1)


nodes = model.Mesh.Nodes;
% from c4d stl out coord to -z-unity coord
nodes(1,:) =  -nodes(1,:);
nodes(2,:) = model.Mesh.Nodes(3,:);
nodes(3,:) = model.Mesh.Nodes(2,:); %

elems = model.Mesh.Elements;
pdeplot3D(model.Mesh)

%%


f = fopen(name + ".txt",'w');
fprintf(f,"%d %d\n", size(nodes,2), size(elems,2));
fprintf(f,"%27.16g %27.16g %27.16g \n", nodes);
fprintf(f,"%d %d %d %d\n", elems);



fclose(f);

%% test


